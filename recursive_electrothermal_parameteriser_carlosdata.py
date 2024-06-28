#!/usr/bin/env python3

from __future__ import annotations
from datetime import datetime
import os
import pandas as pd
import pybop
import fitter

# def parameterise(
#     capacity_Ah: float,
#     data_filename: str,
#     output_filename: str = None,
#     ignore_rests: bool = True,
#     params: dict = BASE_PARAMETER_SET,
#     initial_taus_guess: list[float] = [1, 50],
#     initial_rs_guess: list[float] = [1e-2] * 3,
#     r_bounds: list[float] = [1e-4, 1e-1],
#     c_bounds: list[float] = [1, 1e6],
#     r_variance: float = 1e-3,
#     c_variance: float = 5e2,
#     maxiter=250,
#     method=pybop.XNES,
# ):


def find_txt_files(root_directory: str, recursive=False) -> list[str]:
    ret = []
    if not recursive:
        for file in os.listdir(root_directory):
            if file.endswith(".txt"):
                ret.append(os.path.join(root_directory, file))
        return ret

    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".txt"):
                ret.append(os.path.join(root_directory, file))
    return ret


def get_conditions_from_filename(filename: str) -> tuple[float]:
    # BaSyTec_Kokam_05deg_04_GITT_DsCh_2C.txt
    try:
        chunks = filename.split("_")
        for chunk in chunks:
            if "deg" in chunk:
                temperature = int(chunk[:2])
            if ".txt" in chunk:
                c_rate = int(chunk[0])
        return c_rate, temperature
    except:
        return None, None


def main():
    CAPACITY = 5
    TARGET_C_RATES = [2]
    filenames = find_txt_files("./data")
    parameter_sets = []
    for filename in filenames:
        c_rate, temperature = get_conditions_from_filename(filename)
        if c_rate not in TARGET_C_RATES:
            continue

        print("Starting ", filename)

        filename_stem = "." + filename.split(".")[-2]
        pars_df, ocv_df = fitter.parameterise(
            CAPACITY,
            filename,
            skip_initial_points=0,
            initial_rs_guess=[1e-3] * 3,
            initial_taus_guess=[2, 20],
            r_bounds=[1e-5, 1e-1],
            c_bounds=[1e3, 1e5],
            tau_limits=[5, 50],
            method=pybop.SNES,
            ignore_rests=False,
        )
        pars_df["Temperature_degC"] = temperature
        pars_df["Temperature_degC"] = temperature
        pars_df["Current_A"] = CAPACITY * c_rate

        filename_stem = filename.split(".")[0]
        ocv_df.to_csv(filename_stem + "_ocv.csv", index=False)
        pars_df.to_csv(filename_stem + "_pars.csv", index=False)

        parameter_sets.append(pars_df)
    df = pd.concat(parameter_sets)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv("parameterisation_" + str(now) + ".csv", index=False)


if __name__ == "__main__":
    main()
