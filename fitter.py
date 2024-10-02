#!/usr/bin/env python3

from __future__ import annotations
import copy
import collections
import warnings
import csv
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import pybop
import pybamm

PulseDataset = collections.namedtuple("PulseDataset", ["ts", "vs", "socs", "currents"])

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
    "Initial SoC": None,
}


class ConstrainedThevenin(pybop.empirical.Thevenin):
    def __init__(
        self,
        tau_mins: list | np.ndarray = None,
        tau_maxs: list | np.ndarray = None,
        **model_kwargs,
    ):
        super().__init__(**model_kwargs)
        if tau_maxs is None:
            tau_maxs = [np.inf] * self.pybamm_model.options["number of rc elements"]
        if tau_mins is None:
            tau_mins = [0] * self.pybamm_model.options["number of rc elements"]
        elif (
            len(tau_maxs) != self.pybamm_model.options["number of rc elements"]
            or len(tau_mins) != self.pybamm_model.options["number of rc elements"]
        ):
            raise ValueError(
                "Length of tau constraints must match number of rc elements"
            )
        self._tau_maxs = tau_maxs
        self._tau_mins = tau_mins

    def _check_params(
        self,
        inputs: dict[str, float] = None,
        parameter_set=None,
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


def get_base_parameters(capacity_Ah: float) -> dict:
    pars = copy.deepcopy(BASE_PARAMETER_SET)
    pars["Cell capacity [A.h]"] = capacity_Ah
    return pars


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


def build_ocv_interpolant(socs: np.ndarray, ocvs: np.ndarray) -> pybamm.Interpolant:
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
    integrator_maxstep=None,
) -> tuple[pybop.empirical.Thevenin, pybop.ParameterSet]:
    base_params["Initial SoC"] = initial_soc
    base_params["Open-circuit voltage [V]"] = ocv
    for i in range(n_rc):
        base_params[f"Element-{i+1} initial overpotential [V]"] = 0
        base_params[f"R{i+1} [Ohm]"] = 0.0002  # These should be overwritten
        base_params[f"C{i+1} [F]"] = 1000

    if integrator_maxstep:
        solver = pybamm.ScipySolver(extra_options={"max_step": integrator_maxstep})
    else:
        solver = None
    if tau_maxs is None and tau_mins is None:
        model = pybop.empirical.Thevenin(
            parameter_set=pybop.ParameterSet(params_dict=base_params),
            solver=solver,
            options={"number of rc elements": n_rc},
        )
    else:
        model = ConstrainedThevenin(
            tau_mins,
            tau_maxs,
            parameter_set=pybop.ParameterSet(params_dict=base_params),
            solver=solver,
            options={"number of rc elements": n_rc},
        )
    return model


def get_fitting_params(
    prev_rs: list[float],
    prev_cs: list[float],
    r_bounds: list[float] = [
        1e-4,
        1e-1,
    ],
    c_bounds: list[float] = [1e2, 1e6],
    sigma_r: float = None,
    sigma_c: float = None,
) -> list[pybop.Parameter]:
    """
    TODO check for consistency between initial taus guess, initial rs guess
    """
    if sigma_r:
        prior_r = pybop.Gaussian(prev_rs[0], sigma_r)
    else:
        prior_r = None
    to_fit = [
        pybop.Parameter(
            "R0 [Ohm]",
            initial_value=prev_rs[0],
            prior=prior_r,
            bounds=r_bounds,
        )
    ]
    for i, (prev_r, prev_c) in enumerate(zip(prev_rs[1:], prev_cs)):
        if sigma_c:
            prior_c = pybop.Gaussian(prev_c, sigma_c)
        else:
            prior_c = None
        to_fit.append(
            pybop.Parameter(
                f"C{i+1} [F]",
                initial_value=prev_c,
                prior=prior_c,
                bounds=c_bounds,
            )
        )

        if sigma_r:
            prior_r = pybop.Gaussian(prev_r, sigma_r)
        else:
            prior_r = None
        to_fit.append(
            pybop.Parameter(
                f"R{i+1} [Ohm]",
                initial_value=prev_r,
                prior=prior_r,
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
    scipy_constraints=None,
    p=2,
) -> tuple[np.ndarray, pybop.FittingProblem, float]:
    dataset = pybop.Dataset(
        {
            "Time [s]": data.ts,
            "Current function [A]": data.currents,
            "Voltage [V]": data.vs,
        }
    )
    problem = pybop.FittingProblem(model, fitting_parameters, dataset)
    cost = pybop.SumSquaredError(problem) if p == 2 else pybop.Minkowski(problem, p)
    if scipy_constraints:
        constraints, bounds = scipy_constraints
        optim = pybop.SciPyMinimize(
            cost,
            method=method,
            constraints=constraints,
            bounds=bounds,
        )
    else:
        try:
            optim = pybop.Optimisation(cost, optimiser=method)
        except ValueError as e:
            warnings.warn(f"Something went wrong: {e}")
            return None, None, None
        optim.set_max_iterations(maxiter)
    try:
        params, finalcost = optim.run()
    except ValueError as e:
        # Typically happens when a point is requested outside of the
        # specified bounds; also lets us kill a single optimisation
        warnings.warn(f"Something went wrong: {e}")
        return None, None, None
    return params, problem, finalcost


def get_scipy_constraints(n_rc, method, tau_mins, tau_maxs, r_bounds, c_bounds):
    if method in ["COBYLA", "COBYQA", "SLSQP", "trust-constr"]:
        # Nonlinear constraints on tau
        def calculate_taus(x):
            return x[1::2] * x[2::2]

        constraint = scipy.optimize.NonlinearConstraint(
            calculate_taus,
            tau_mins,
            tau_maxs,
        )
    else:
        constraint = None

    # Where R0, Ri, Ci lie in the list of fitted parameters
    # 0, 2, 4, 6, ...
    rs_idx = np.arange(n_rc + 1) * 2
    # 1, 3, 5, ...
    cs_idx = np.arange(n_rc) * 2 + 1
    # Bounds arrays, to be filled
    lb = np.zeros(2 * n_rc + 1)
    ub = np.zeros(2 * n_rc + 1)
    # Place Ri, Ci bounds into their respective places of the bounds arrays
    lb[rs_idx] = r_bounds[0]
    lb[cs_idx] = c_bounds[0]
    ub[rs_idx] = r_bounds[1]
    ub[cs_idx] = c_bounds[1]
    bounds = scipy.optimize.Bounds(lb, ub, True)
    return constraint, bounds


def parameterise(
    datasets: PulseDataset | list[PulseDataset],
    ocv_func,
    base_parameters: dict,
    initial_taus_guess: list[float] = [1, 50],
    initial_rs_guess: list[float] = [1e-2] * 3,
    r_bounds: list[float] = [0, np.inf],
    c_bounds: list[float] = [0, np.inf],
    tau_mins: list[float] = None,
    tau_maxs: list[float] = None,
    sigma_r: float | list[float] = None,
    sigma_c: float | list[float] = None,
    maxiter=50,
    integrator_maxstep=None,
    method=pybop.XNES,
    verbose=True,
    p=2,
    plot=True,
):
    n_rc = len(initial_taus_guess)
    if isinstance(datasets, PulseDataset):
        datasets = [datasets]
    if isinstance(method, str) or not hasattr(method, "__len__"):
        methodlist = [method]
    else:
        methodlist = method
    methodcounts = {m: 0 for m in methodlist}

    params = []
    average_socs = []
    for i, dataset in enumerate(datasets):
        initial_soc = dataset.socs[0]
        best_cost = np.inf
        best_pars = None
        best_problem = None
        best_method = None
        for method in methodlist:
            # TODO maybe print method / cost pairs, for validation?
            if isinstance(method, str):
                # Don't get model to apply constraints with constrained optimisers
                model = get_model(
                    initial_soc,
                    ocv_func,
                    base_parameters,
                    n_rc,
                    integrator_maxstep=integrator_maxstep,
                )
            else:
                model = get_model(
                    initial_soc,
                    ocv_func,
                    base_parameters,
                    n_rc,
                    tau_maxs,
                    tau_mins,
                    integrator_maxstep,
                )

            if len(params) == 0:
                prev_rs = initial_rs_guess
                prev_cs = [
                    tau / r for tau, r in zip(initial_taus_guess, initial_rs_guess[1:])
                ]
            else:
                prev_rs = params[-1][::2]
                prev_cs = params[-1][1::2]

            if isinstance(method, str):
                scipy_constraints = get_scipy_constraints(
                    n_rc, method, tau_mins, tau_maxs, r_bounds, c_bounds
                )
            else:
                scipy_constraints = None

            fitting_params = get_fitting_params(
                prev_rs, prev_cs, r_bounds, c_bounds, sigma_r, sigma_c
            )
            fitted, problem, finalcost = fit_parameter_set(
                dataset,
                model,
                fitting_params,
                maxiter,
                method,
                scipy_constraints,
                p,
            )
            if fitted is None:
                continue

            if finalcost < best_cost:
                best_cost = finalcost
                best_pars = fitted
                best_problem = problem
                best_method = method

        if np.isinf(best_cost):
            continue
        params.append(best_pars)
        average_socs.append(np.mean(dataset.socs))
        methodcounts[best_method] += 1

        if verbose:
            print_params(best_pars)
            print(f"Best method: {best_method}")
            print(f"Final cost: {best_cost}")
        if plot:
            pybop.quick_plot(best_problem, problem_inputs=best_pars)

    names = ["R0"]
    for i in range(n_rc):
        names.append(f"C{i+1}")
        names.append(f"R{i+1}")
    ret_df = pd.DataFrame(params, columns=names)
    ret_df.insert(0, "SOC", average_socs)

    if verbose:
        print("Finished parameterising; best optimisation methods:\n", methodcounts)

    return ret_df
