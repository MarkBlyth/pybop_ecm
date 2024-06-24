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


class ConstrainedThevenin(pybop.empirical.Thevenin):
    def __init__(self, tau_limits: list | np.ndarray, **model_kwargs):
        super().__init__(**model_kwargs)
        if tau_limits is None:
            tau_limits = [np.inf] * self.pybamm_model.options["number of rc elements"]
        elif len(tau_limits) != self.pybamm_model.options["number of rc elements"]:
            raise ValueError(
                "Length of tau constraints must match number of rc elements"
            )
        self._tau_limits = tau_limits

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
                if tau > self._tau_limits[i - 1]:
                    return False
                i += 1
            else:
                return True


PulseDataset = collections.namedtuple("PulseDataset", ["ts", "vs", "socs", "currents"])

BASE_PARAMETER_SET = {
    "chemistry": "ecm",
    "Initial temperature [K]": 25 + 273.15,
    "Upper voltage cut-off [V]": 4.25,
    "Lower voltage cut-off [V]": 2.5,
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


def get_discharge_pulse_data(
    df: pd.DataFrame,
    socs: np.ndarray,
    ignore_rests: bool = False,
    skip_initial_points: int = 0,
) -> list[PulseDataset]:
    end_of_rests = df[
        df["Command"].eq("Pause") & df.shift(-1)["Command"].eq("Discharge")
    ]
    ret = []
    for start, end in zip(end_of_rests.index, end_of_rests.index[1:]):
        pulse_df = df.iloc[start:end]
        if ignore_rests:
            pulse_df = pulse_df[pulse_df["Command"].ne("Pause")]
        soclist = socs[pulse_df.index]
        dataset = PulseDataset(
            pulse_df["~Time[s]"].to_numpy()[skip_initial_points:],
            pulse_df["U[V]"].to_numpy()[skip_initial_points:],
            soclist[skip_initial_points:],
            -pulse_df["I[A]"].to_numpy()[skip_initial_points:],
        )
        ret.append(dataset)
    return ret


def get_ocvs(df: pd.DataFrame, socs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # TODO instead, run from last data of each pulse, to avoid re-searching for end of pulses
    end_of_rests = df[
        df["Command"].eq("Pause") & df.shift(-1)["Command"].eq("Discharge")
    ]
    vs = end_of_rests["U[V]"].to_numpy()
    # _, ax = plt.subplots()
    # ax.plot(df["~Time[s]"], df["U[V]"])
    # ax.scatter(end_of_rests["~Time[s]"], end_of_rests["U[V]"])
    # plt.show()

    # TODO THIS MISSES OUT THE VERY LAST PULSE!!!
    return socs[end_of_rests.index], vs


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
    tau_limits=None,
) -> tuple[pybop.empirical.Thevenin, pybop.ParameterSet]:
    base_params["Initial SoC"] = initial_soc
    base_params["Open-circuit voltage [V]"] = ocv
    for i in range(n_rc):
        base_params[f"Element-{i+1} initial overpotential [V]"] = 0
        base_params[f"R{i+1} [Ohm]"] = 0.0002
        base_params[f"C{i+1} [F]"] = 1000
    # ...and if there's issues, also...
    # "R1 [Ohm]": 0.0002,
    # "C1 [F]": 10000,

    model = ConstrainedThevenin(
        tau_limits, parameter_set=base_params, options={"number of rc elements": n_rc}
    )
    return model


def get_fitting_params(
    prev_rs: np.ndarray,
    prev_cs: np.ndarray,
    r_bounds: list[float] = [
        1e-4,
        1e-1,
    ],  ######### These set the bounds on R, C
    c_bounds: list[float] = [1e2, 1e6],
    r_variance: float = 1e-4,  ############ These two dictate how similar the next solution looks to the previous one
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


def fit_parameter_set(
    data: PulseDataset,
    model: pybop.empirical.Thevenin,
    fitting_parameters: list[pybop.Parameter],
    maxiter=50,
    method=pybop.SNES,
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
    print_params(params)
    pybop.quick_plot(problem, parameter_values=params)
    return params


# Make usable


def print_params(params: np.ndarray):
    print("R0: ", params[0])
    for i, (ri, ci) in enumerate(zip(params[2::2], params[1::2])):
        print(f"R{i+1}: {ri}, C{i+1}: {ci}, tau{i+1}: {ri*ci}")


def _get_header_line_number(filename):
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if row[0][0] != "~":
                return max(i - 1, 0)
    return 0


def import_data(filename: str) -> pd.DataFrame:
    header_line = _get_header_line_number(filename)
    # return pd.read_csv(filename, header=header_line, encoding_errors="ignore")
    return pd.read_csv(
        filename, header=header_line, sep="\s+", encoding="unicode_escape"
    )


def parameterise(
    capacity_Ah: float,
    data_filename: str,
    output_filename: str = None,
    ignore_rests: bool = True,  ####################### If true, only fit to I!=0; recommended, for more robust fitting
    skip_initial_points: int = 0,
    base_parameters: dict = BASE_PARAMETER_SET,
    initial_taus_guess: list[float] = [1, 50],
    initial_rs_guess: list[float] = [1e-2] * 3,
    r_bounds: list[float] = [1e-4, 1e-1],
    c_bounds: list[float] = [1, 1e6],
    tau_limits: list[float] = None,
    r_variance: float = 1e-3,
    c_variance: float = 5e2,
    maxiter=250,
    method=pybop.XNES,
):
    base_parameters["Cell capacity [A.h]"] = capacity_Ah
    n_rc = len(initial_taus_guess)
    df = import_data(data_filename)

    socs = coulomb_count(
        df["~Time[s]"].to_numpy(),
        df["I[A]"].to_numpy(),
        capacity_Ah,
    )
    ocvdata_socs, ocvdata_vs = get_ocvs(df, socs)
    ocv_func = build_ocv_interpolant(ocvdata_socs, ocvdata_vs)

    ds_pulses = get_discharge_pulse_data(df, socs, ignore_rests, skip_initial_points)

    params = []
    average_socs = []
    for i, (pulse, initial_soc) in enumerate(zip(ds_pulses, ocvdata_socs)):
        model = get_model(initial_soc, ocv_func, base_parameters, n_rc, tau_limits)
        if len(params) == 0:
            prev_rs = initial_rs_guess
            prev_cs = [
                tau / r for tau, r in zip(initial_taus_guess, initial_rs_guess[1:])
            ]
        else:
            prev_rs = params[-1][::2]
            prev_cs = params[-1][1::2]
        fitting_params = get_fitting_params(
            prev_rs, prev_cs, r_bounds, c_bounds, r_variance, c_variance
        )
        params.append(fit_parameter_set(pulse, model, fitting_params, maxiter, method))
        average_socs.append(np.mean(pulse.socs))

    names = ["R0"]
    for i in range(n_rc):
        names.append(f"C{i+1}")
        names.append(f"R{i+1}")
    ret_df = pd.DataFrame(params, columns=names)
    ret_df.insert(0, "SOC", average_socs)

    ocv_df = pd.DataFrame.from_dict({"SOC": ocvdata_socs, "OCV": ocvdata_vs})
    if output_filename is not None:
        ret_df.to_csv(output_filename, index=False)
        ocv_df.to_csv("ocv_" + output_filename, index=False)
    return ret_df, ocv_df


def main():
    # TODO 1. parameterise in both ds and ch; 2. add current-draw to resulting LUT
    # TODO strip off leading ~ s from BaSyTec files
    pars = parameterise(5, "GITT.csv")


if __name__ == "__main__":
    main()
