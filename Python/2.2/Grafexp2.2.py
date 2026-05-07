from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


X_COLUMN = "Time (s)"
Y_COLUMN = "Angular Velocity (rad/s)"

START_TIME = 0
END_TIME = 4.0

EXPERIMENT_NAME = Path(__file__).parent.name
CSV_FILES = sorted(Path(__file__).parent.glob(f"{EXPERIMENT_NAME}_run_*.csv"))
TIME_LABEL = f"{START_TIME:g}-{END_TIME:g} s"


def read_measurements(csv_file):
    df = pd.read_csv(csv_file, sep=";", decimal=",", encoding="utf-8-sig")
    df = df[(df[X_COLUMN] >= START_TIME) & (df[X_COLUMN] <= END_TIME)]
    return df[[X_COLUMN, Y_COLUMN]].dropna()


all_x = []
all_y = []

plt.figure(figsize=(9, 6))
markers = ["o", "s", "^", "D", "v"]

for csv_file, marker in zip(CSV_FILES, markers):
    data = read_measurements(csv_file)
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
        label=csv_file.stem,
    )

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

if len(all_x) == 0:
    raise ValueError(f"Inga datapunkter hittades i intervallet {TIME_LABEL}.")

k, m = np.polyfit(all_x, all_y, 1)
x_line = np.linspace(all_x.min(), all_x.max(), 200)
y_line = k * x_line + m

plt.plot(
    x_line,
    y_line,
    color="black",
    linewidth=2,
    label=f"Gemensam regression: y = {k:.4g}x + {m:.4g}",
)

plt.xlabel(X_COLUMN)
plt.ylabel(Y_COLUMN)
plt.title(f"{EXPERIMENT_NAME}: mätpunkter {TIME_LABEL}, k = {k:.6g}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim(START_TIME, END_TIME)
plt.show()

print(f"Lutning k = {k:.6g}")
print(f"Intercept m = {m:.6g}")
