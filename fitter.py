#!/usr/bin/env python3

from __future__ import annotations
import collections
import warnings
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybop
import pybamm

PulseDataset = collections.namedtuple(
    "PulseDataset", ["ts", "vs", "socs", "currents"]
)

BASE_PARAMETER_SET = {
    "chemistry": "ecm",
    "Initial temperature [K]": 25 + 273.15,
    "Upper voltage cut-off [V]": np.inf,
    "Lower voltage cut-off [V]": -np.inf,
    "Nominal cell capacity [A.h]": 5,
    "Ambient temperature [K]": 25 + 273.15,
    "Current function [A]": 5,
    "R0 [Ohm]": 0.001,
    "Cell thermal mass [J/K]": 1000,
    "Cell-jig heat transfer coefficient [W/K]": 10,
    "Jig thermal mass [J/K]": 500,
    "Jig-air heat transfer coefficient [W/K]": 10,
    "Entropic change [V/K]": 0.0004,
}


class ConstrainedThevenin(pybop.empirical.Thevenin):
    def __init__(self, tau_mins: list | np.ndarray = None, tau_maxs: list | np.ndarray = None, **model_kwargs):
        super().__init__(**model_kwargs)
        if tau_maxs is None:
            tau_maxs = [np.inf] * self.pybamm_model.options[
                "number of rc elements"
            ]
        if tau_mins is None:
            tau_mins = [0] * self.pybamm_model.options[
                "number of rc elements"
            ]
        elif (
            len(tau_maxs)
            != self.pybamm_model.options["number of rc elements"]
            or len(tau_mins)
            != self.pybamm_model.options["number of rc elements"]
        ):
            raise ValueError(
                "Length of tau constraints must match number of rc elements"
            )
        self._tau_maxs = tau_maxs
        self._tau_mins = tau_mins

    def _check_params(
        self,
        inputs: dict[str, float] = None,
        allow_infeasible_solutions: bool = False,
    ) -> bool:
        # Check every respective R*C <= tau_bound

        i = 1
        if inputs is None:
            # Simulating the model will result in this being called with
            # inputs=None; must return true to allow the simulation to run
            return True
        while True:
            if f"C{i} [F]" in inputs and f"R{i} [Ohm]" in inputs:
                tau = inputs[f"R{i} [Ohm]"] * inputs[f"C{i} [F]"]
                if tau > self._tau_maxs[i - 1]:
                    return False
                if tau < self._tau_mins[i - 1]:
                    return False
                i += 1
            else:
                return True


# Handle data


def coulomb_count(
    ts: np.ndarray,
    currents: np.ndarray,
    capacity: float,
    initial_soc: float = 1,
) -> np.ndarray:
    if currents.shape != ts.shape:
        raise ValueError("Current and ts must have same shape")
    ret = np.zeros_like(currents)
    ret[0] = 0
    ret[1:] = np.diff(ts) * currents[:-1]
    return np.cumsum(ret) / (capacity * 3600) + initial_soc


def build_ocv_interpolant(
    socs: np.ndarray, ocvs: np.ndarray
) -> pybamm.Interpolant:
    idxs = np.argsort(socs)

    def ocv(soc):
        return pybamm.Interpolant(socs[idxs], ocvs[idxs], soc, "OCV(SOC)")

    return ocv


# Run parameterisation step


def get_model(
    initial_soc,
    ocv,
    base_params,
    n_rc: int = 2,
    tau_maxs=None,
    tau_mins=None,
) -> tuple[pybop.empirical.Thevenin, pybop.ParameterSet]:
    base_params["Initial SoC"] = initial_soc
    base_params["Open-circuit voltage [V]"] = ocv
    for i in range(n_rc):
        base_params[f"Element-{i+1} initial overpotential [V]"] = 0
        base_params[f"R{i+1} [Ohm]"] = 0.0002  # These should be overwritten
        base_params[f"C{i+1} [F]"] = 1000

    model = ConstrainedThevenin(
        tau_mins,
        tau_maxs,
        parameter_set=base_params,
        options={"number of rc elements": n_rc},
    )
    return model


def get_fitting_params(
    prev_rs: np.ndarray,
    prev_cs: np.ndarray,
    r_bounds: list[float] = [
        1e-4,
        1e-1,
    ],
    c_bounds: list[float] = [1e2, 1e6],
    r_variance: float = 1e-4,
    c_variance: float = 1e3,
) -> list[pybop.Parameter]:
    to_fit = [
        pybop.Parameter(
            "R0 [Ohm]",
            prior=pybop.Gaussian(prev_rs[0], r_variance),
            bounds=r_bounds,
        )
    ]
    for i, (prev_r, prev_c) in enumerate(zip(prev_rs[1:], prev_cs)):
        to_fit.append(
            pybop.Parameter(
                f"C{i+1} [F]",
                prior=pybop.Gaussian(prev_c, c_variance),
                bounds=c_bounds,
            )
        )
        to_fit.append(
            pybop.Parameter(
                f"R{i+1} [Ohm]",
                prior=pybop.Gaussian(prev_r, r_variance),
                bounds=r_bounds,
            )
        )
    return to_fit


def print_params(params: np.ndarray):
    print("R0: ", params[0])
    for i, (ri, ci) in enumerate(zip(params[2::2], params[1::2])):
        print(f"R{i+1}: {ri}, C{i+1}: {ci}, tau{i+1}: {ri*ci}")


def fit_parameter_set(
    data: PulseDataset,
    model: pybop.empirical.Thevenin,
    fitting_parameters: list[pybop.Parameter],
    maxiter=50,
    method=pybop.XNES,
) -> np.ndarray:
    dataset = pybop.Dataset(
        {
            "Time [s]": data.ts,
            "Current function [A]": data.currents,
            "Voltage [V]": data.vs,
        }
    )
    problem = pybop.FittingProblem(model, fitting_parameters, dataset)
    cost = pybop.SumSquaredError(problem)
    optim = pybop.Optimisation(cost, optimiser=method)
    optim.set_max_iterations(maxiter)
    params, finalcost = optim.run()
    return params, problem, finalcost


def parameterise(
    datasets: PulseDataset | list[PulseDataset],
    ocv_func,
    base_parameters: dict,
    initial_taus_guess: list[float] = [1, 50],
    initial_rs_guess: list[float] = [1e-2] * 3,
    r_bounds: list[float] = [1e-4, 1e-1],
    c_bounds: list[float] = [1, 1e6],
    tau_maxs: list[float] = None,
    tau_mins: list[float] = None,
    r_variance: float = 1e-3,
    c_variance: float = 5e2,
    maxiter=50,
    method=pybop.XNES,
    verbose=True,
    plot=True,
):
    n_rc = len(initial_taus_guess)
    if isinstance(datasets, PulseDataset):
        datasets = [datasets]

    params = []
    average_socs = []
    for i, dataset in enumerate(datasets):
        initial_soc = dataset.socs[0]
        model = get_model(
            initial_soc, ocv_func, base_parameters, n_rc, tau_maxs, tau_mins,
        )
        if len(params) == 0:
            prev_rs = initial_rs_guess
            prev_cs = [
                tau / r
                for tau, r in zip(initial_taus_guess, initial_rs_guess[1:])
            ]
        else:
            prev_rs = params[-1][::2]
            prev_cs = params[-1][1::2]
        fitting_params = get_fitting_params(
            prev_rs, prev_cs, r_bounds, c_bounds, r_variance, c_variance
        )
        fitted, problem, finalcost = fit_parameter_set(
            dataset, model, fitting_params, maxiter, method
        )
        params.append(fitted)
        average_socs.append(np.mean(dataset.socs))

        if verbose:
            print_params(fitted)
            print(f"Final cost: {finalcost}")
        if plot:
            pybop.quick_plot(problem, parameter_values=fitted)

    names = ["R0"]
    for i in range(n_rc):
        names.append(f"C{i+1}")
        names.append(f"R{i+1}")
    ret_df = pd.DataFrame(params, columns=names)
    ret_df.insert(0, "SOC", average_socs)

    return ret_df
