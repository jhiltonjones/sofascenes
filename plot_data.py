import os
import csv
import math

import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "force_log.csv")
OUT_PNG = os.path.join(HERE, "force_magnitude.png")


def read_force_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    t = []
    F = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "t_s" not in cols:
            raise ValueError(f"Expected column 't_s' in CSV. Found: {cols}")

        # Prefer Newtons if available; otherwise fall back to mN
        has_F_N = "Fmag_N" in cols
        has_F_mN = "Fmag_mN" in cols
        if not has_F_N and not has_F_mN:
            raise ValueError("Expected 'Fmag_N' or 'Fmag_mN' column in CSV.")

        for row in reader:
            try:
                ts = float(row["t_s"])
            except (TypeError, ValueError):
                continue

            key = "Fmag_N" if has_F_N else "Fmag_mN"
            try:
                val = float(row[key])
            except (TypeError, ValueError):
                continue

            # Skip NaNs
            if math.isnan(ts) or math.isnan(val):
                continue

            # If using mN, convert to N for plotting consistency
            if key == "Fmag_mN":
                val *= 1e-3

            t.append(ts)
            F.append(val)

    return t, F


def main():
    t, F = read_force_csv(CSV_PATH)

    if len(t) == 0:
        raise RuntimeError("No valid samples found in CSV (all NaN/empty?).")

    plt.figure()
    plt.plot(t, F)
    plt.xlabel("Time (s)")
    plt.ylabel("|F| (N)")
    plt.title("Tip contact force magnitude")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved plot to: {OUT_PNG}")

    # Optional: show interactively
    plt.show()


if __name__ == "__main__":
    main()
