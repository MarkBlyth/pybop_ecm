import csv
import warnings
from dataclasses import dataclass
import pandas as pd
import numpy as np
from fitter import PulseDataset


@dataclass
class Headers:
    command: str
    time: str
    current: str
    voltage: str
    charging: str
    discharging: str
    resting: str


BasytecHeaders = Headers(
    "Command", "~Time[s]", "I[A]", "U[V]", "Charge", "Discharge", "Pause"
)
NewareHeaders = Headers(
    "Step Type", "~Time[s]", "I[A]", "U[V]", "CC Chg", "CC DChg", "Rest"
)  #### TODO check time; I; V


def _get_basytec_header_line_number(filename):
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if row[0][0] != "~":
                return max(i - 1, 0)
    return 0


def import_data(filename: str, cycler: str) -> pd.DataFrame:
    if cycler == "neware":
        return import_neware(filename)
    if cycler == "basytec":
        return import_basytec(filename)
    raise ValueError(
        "Cycler must be basytec or neware, but received {cycler}"
    )


def import_basytec(filename: str) -> pd.DataFrame:
    header_line = _get_basytec_header_line_number(filename)
    return pd.read_csv(  ############## TODO HAVE ENCODING OPTION (tabs / comma separators)
        filename, header=header_line, sep="\s+", encoding="unicode_escape"
    )


def import_neware(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, encoding_errors="ignore")
    seconds = (
        df[NewareHeaders.time]
        .str.split(":")
        .apply(lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
    )
    df[NewareHeaders.time] = seconds.to_numpy()
    return df


def get_pulse_data(
    df: pd.DataFrame,
    socs: np.ndarray,
    headers: Headers,
    direction: str,
    ignore_rests: bool = False,
    skip_initial_points: int = 0,
) -> list[PulseDataset]:
    if direction not in ["charge", "discharge", "switch"]:
        raise ValueError(
            f"direction must be charge, discharge, or switch; received {direction}"
        )
    active_command = (
        headers.charging if direction == "charge" else headers.discharging
    )
    wrong_command = (
        headers.discharging if direction == "charge" else headers.charging
    )
    warnings.warn("This pulse-getter may miss out the last pulse in a GITT")

    end_of_rests = df[
        df[headers.command].eq(headers.resting)
        & df.shift(-1)[headers.command].eq(active_command)
    ]
    ret = []
    for start, end in zip(end_of_rests.index, end_of_rests.index[1:]):
        pulse_df = df.iloc[start:end]
        if any(pulse_df[headers.command].eq(wrong_command)):
            if direction != "switch":
                continue
            if not any(pulse_df[headers.command].eq(active_command)):
                continue
        elif direction == "switch":
            continue
        if ignore_rests:
            pulse_df = pulse_df[
                pulse_df[headers.command].ne(headers.resting)
            ]
        soclist = socs[pulse_df.index]

        unique_times = np.r_[True, np.diff(pulse_df[headers.time]) != 0]
        if not all(unique_times):
            warnings.warn(
                f"Skipping double-counted time-sample \n{pulse_df[np.logical_not(unique_times)]}"
            )
            pulse_df = pulse_df[unique_times]
            soclist = soclist[unique_times]

        dataset = PulseDataset(
            pulse_df["~Time[s]"].to_numpy()[skip_initial_points:],
            pulse_df["U[V]"].to_numpy()[skip_initial_points:],
            soclist[skip_initial_points:],
            -pulse_df["I[A]"].to_numpy()[skip_initial_points:],
        )
        ret.append(dataset)
    return ret


def get_ocvs_from_df(
    df: pd.DataFrame,
    socs: np.ndarray,
    headers: Headers,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    active_command = (
        headers.charging if direction == "charge" else headers.discharging
    )
    end_of_rests = df[
        df[headers.command].eq(headers.resting)
        & df.shift(-1)[headers.command].eq(active_command)
    ]
    vs = end_of_rests[headers.voltage].to_numpy()

    warnings.warn("This OCV-getter may miss out the last pulse in a GITT")

    return socs[end_of_rests.index], vs


def get_ocvs_from_pulsedataset_list(
    pulses: list[PulseDataset],
) -> tuple[np.ndarray, np.ndarray]:
    socs, vs = np.zeros(len(pulses)), np.zeros(len(pulses))
    for i, pulse in enumerate(pulses):
        is_resting = pulse.currents == 0

        # Find the longest continuous set of rests, to make sure we're picking out the right data
        # https://stackoverflow.com/questions/56301970/how-can-i-return-the-longest-continuous-occurrence-of-true-in-boolean-and-rep
        bools = np.r_[False, is_resting, False]
        # Get indices of group shifts
        shiftpoints = np.flatnonzero(bools[:-1] != bools[1:])
        # Get group lengths and hence the max index group
        groupsizes = (shiftpoints[1::2] - shiftpoints[::2]).argmax()
        # Initialize array and assign only the largest True island as True.
        out = np.zeros_like(is_resting)
        out[
            shiftpoints[2 * groupsizes] : shiftpoints[2 * groupsizes + 1]
        ] = 1

        # Get last datapoint from longest continuous rest
        rest_soc = pulse.socs[out][-1]
        rest_v = pulse.vs[out][-1]

        socs[i] = rest_soc
        vs[i] = rest_v
    ordering = np.argsort(socs)
    return socs[ordering], vs[ordering]
