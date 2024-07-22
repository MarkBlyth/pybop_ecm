#!/usr/bin/env python3

from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pybop
import fitter
import datareaders


def main():
    capacity_Ah = 2.16
    base_params = fitter.get_base_parameters(capacity_Ah)

    df = datareaders.import_neware("MLP001.csv")

    socs = fitter.coulomb_count(
        df[datareaders.BasytecHeaders.time],
        df[datareaders.BasytecHeaders.current],
        capacity_Ah,
        0.98,
    )
    pulses = datareaders.get_pulse_data(
        df, socs, datareaders.NewareHeaders, "switch"
    )[1: -1]

    ocv_socs, ocv_vs = datareaders.get_ocvs_from_pulsedataset_list(pulses)
    warnings.warn(
        "Adding fictitious OCV points outside cell operating voltages, for interpolator stability"
    )
    ocv_socs = np.r_[-0.01, ocv_socs, 1.01]
    ocv_vs = np.r_[2.99, ocv_vs, 4.21]
    ocv_func = fitter.build_ocv_interpolant(ocv_socs, ocv_vs)

    pars_df = fitter.parameterise(
        pulses,
        ocv_func,
        base_params,
        initial_taus_guess=[10, 70],
        tau_mins=[0, 0],
        tau_maxs=[25, 160],
        r_bounds=[5e-3, 1e-1],
        c_bounds=[1, 1e6],
        sigma_r=0.01,
        sigma_c=500,
        initial_rs_guess=[0.05, 0.01, 0.01],
        method="SLSQP",
        integrator_maxstep=10,
        plot=True,
    )
    pars_df["Temperature_degC"] = 25
    ocv_df = pd.DataFrame.from_dict({"SOC": ocv_socs, "OCV[V]": ocv_vs})

    ocv_df.to_csv("mlp001_25_ocv.csv", index=False)
    pars_df.to_csv("mlp001_25_pars.csv", index=False)


if __name__ == "__main__":
    main()
