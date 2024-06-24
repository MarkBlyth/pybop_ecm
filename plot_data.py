import csv
import pandas as pd
import matplotlib.pyplot as plt

FILE = "GITT.csv"

class BasytecHeaders:
    CURRENT_HEADER = "I[A]"
    VOLTAGE_HEADER = "U[V]"
    TIME_HEADER = "~Time[s]"
    SOC_HEADER = "Coulomb Counting SoC (%)"
    COMMAND_HEADER = "Command"
    PAUSE_COMMAND = "Pause"
    DISCHARGE_COMMAND = "Discharge"
    CAPACITY_DICT_KEY = "~Nominal battery capacity [Ah]"


def _get_header_line_number(filename):
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if row[0][0] != "~":
                return max(i - 1, 0)
    return 0


def load_csv_to_dataframe(filename):
    header_line = _get_header_line_number(filename)
    return pd.read_csv(filename, header=header_line, sep="\s+", encoding="unicode_escape")


def load_preamble_to_dict(filename):
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        ret_dict = {}
        for i, row in enumerate(f):
            try:
                splitrow = row.split(": ")
                val = ": ".join(splitrow[1:])
                try:
                    ret_dict[splitrow[0]] = float(val)
                except ValueError:
                    ret_dict[splitrow[0]] = val
            except ValueError:
                pass
            if row[0][0] != "~":
                break
    return ret_dict


def main():
    df = load_csv_to_dataframe(FILE)
    ts = df["~Time[s]"].to_numpy()
    vs = df["U[V]"].to_numpy()
    _, ax = plt.subplots()
    ax.plot(ts, vs)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    plt.show()

if __name__ == "__main__":
    main()