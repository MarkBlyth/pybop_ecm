#!/usr/bin/env python3

from __future__ import annotations
import os
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import pybop
import fitter
import datareaders


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
    capacity_Ah = 5
    base_params = fitter.get_base_parameters(capacity_Ah)
    TARGET_C_RATES = [2]

    filenames = find_txt_files("./data")
    parameter_sets = []
    for filename in filenames:
        c_rate, temperature = get_conditions_from_filename(filename)
        if c_rate not in TARGET_C_RATES:
            continue

        filename_stem = "." + filename.split(".")[-2]
        print("Starting ", filename)

        df = datareaders.import_basytec(filename)

        socs = fitter.coulomb_count(
            df[datareaders.BasytecHeaders.time],
            df[datareaders.BasytecHeaders.current],
            capacity_Ah,
            1,
        )

        pulses = datareaders.get_pulse_data(
            df, socs, datareaders.BasytecHeaders, "charge"
        )
        ocv_socs, ocv_vs = datareaders.get_ocvs_from_pulsedataset_list(pulses)
        warnings.warn(
            "Adding fictitious OCV points outside cell operating voltages, for interpolator stability"
        )
        ocv_socs = np.r_[-0.01, ocv_socs, 1.01]
        ocv_vs = np.r_[2.49, ocv_vs, 4.21]
        ocv_func = fitter.build_ocv_interpolant(ocv_socs, ocv_vs)

        pulses = datareaders.get_pulse_data(
            df, socs, datareaders.BasytecHeaders, "charge", ignore_rests=True, skip_initial_points=2,
        )

        pars_df = fitter.parameterise(
            pulses,
            ocv_func,
            base_params,
            initial_taus_guess=[6, 13],
            tau_mins=[1, 1],
            tau_maxs=[200, 200],
            r_bounds=[0, 1e-1],
            c_bounds=[1e1, 1e6],
            r_variance=1e-5,
            c_variance=1e-2,
            initial_rs_guess=[0.005, 0.01, 0.01],
            maxiter=1000,
            method=pybop.XNES,
            plot=True,
        )
        pars_df["Temperature_degC"] = temperature
        pars_df["Current_A"] = capacity_Ah * c_rate
        ocv_df = pd.DataFrame.from_dict({"SOC": ocv_socs, "OCV[V]": ocv_vs})

        ocv_df.to_csv(str(temperature) + "degC_ocv.csv", index=False)
        pars_df.to_csv(str(temperature) + "degC_pars.csv", index=False)

        parameter_sets.append(pars_df)
    df = pd.concat(parameter_sets)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv("parameterisation_" + str(now) + ".csv", index=False)


if __name__ == "__main__":
    main()
