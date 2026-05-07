from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


X_COLUMN = "Time (s)"
Y_COLUMN = "Angular Velocity (rad/s)"

TIME_INTERVALS = {
    "2.1_run_1": (1.0, 1.5),
    "2.1_run_2": (1.0, 1.5),
    "2.1_run_3": (1.0, 1.5),
    "2.1_run_4": (1.0, 1.5),
    "2.1_run_5": (1.0, 1.5),
}

CSV_FILES = sorted(Path(__file__).parent.glob("2.1_run_*.csv"))


def read_measurements(csv_file):
    df = pd.read_csv(csv_file, sep=";", decimal=",")
    start_time, end_time = TIME_INTERVALS[csv_file.stem]
    df = df[(df[X_COLUMN] >= start_time) & (df[X_COLUMN] <= end_time)]
    return df[[X_COLUMN, Y_COLUMN]].dropna(), start_time, end_time

all_x = []
all_y = []

plt.figure(figsize=(9, 6))
markers = ["o", "s", "^", "D", "v"]

for csv_file, marker in zip(CSV_FILES, markers):
    data, start_time, end_time = read_measurements(csv_file)
    x = data[X_COLUMN].to_numpy()
    y = data[Y_COLUMN].to_numpy()

    all_x.append(x)
    all_y.append(y)

    plt.plot(
        x,
        y,
        linestyle="None",
        marker=marker,
        markersize=4,
        alpha=0.75,
        label=f"{csv_file.stem}, {start_time:g}-{end_time:g} s",
    )

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

if len(all_x) == 0:
    raise ValueError("Inga datapunkter hittades i de valda tidsintervallen.")

k, m = np.polyfit(all_x, all_y, 1)
x_line = np.linspace(all_x.min(), all_x.max(), 200)
y_line = k * x_line + m

plt.plot(
    x_line,
    y_line,
    color="black",
    linewidth=2,
    label=f"Common linear regression: y = {k:.4g}x + {m:.4g}",
)

plt.xlabel(X_COLUMN)
plt.ylabel(Y_COLUMN)
plt.title(f"Disk + Ring, k = {k:.6g}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim(all_x.min(), all_x.max())
plt.show()

print(f"Lutning k = {k:.6g}")
print(f"Intercept m = {m:.6g}")
