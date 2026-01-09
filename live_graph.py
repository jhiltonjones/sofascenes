import os
import time
import numpy as np
import pandas as pd

os.environ["MPLBACKEND"] = "QtAgg"
import matplotlib.pyplot as plt

CSV = "/home/jack/sofascenes/contact_angle5.csv"

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="polar")
sc = ax.scatter([], [], s=5)
ax.set_title("Contact angle around circumference (radius = time)")
ax.set_rlabel_position(135)
ax.set_theta_zero_location("N")   # 0° at the right (East)  -> bottom will be 270°
ax.set_theta_direction(-1)         # counter-clockwise (default)


fig.show()

last_good = None
last_offsets = None
last_rows = 0

def read_csv_robust(path: str) -> pd.DataFrame | None:
    try:
        # engine="python" tends to be more tolerant during concurrent writes
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    except Exception:
        return None

    if not {"t", "theta_wrapped"}.issubset(df.columns):
        return None

    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["theta_wrapped"] = pd.to_numeric(df["theta_wrapped"], errors="coerce")
    if "has_contact" in df.columns:
        df["has_contact"] = pd.to_numeric(df["has_contact"], errors="coerce")

    return df

while True:
    try:
        df = read_csv_robust(CSV)
        if df is not None and len(df) > 0:
            last_good = df

        if last_good is not None:
            df = last_good

            if "has_contact" in df.columns:
                mask = (df["has_contact"] == 1) & np.isfinite(df["theta_wrapped"]) & np.isfinite(df["t"])
            else:
                mask = np.isfinite(df["theta_wrapped"]) & np.isfinite(df["t"])

            theta = df.loc[mask, "theta_wrapped"].to_numpy(float)
            r = df.loc[mask, "t"].to_numpy(float)
            rmax = float(np.nanmax(df["t"].to_numpy(float)))  # or np.nanmax(r) if you prefer contact-only
            if np.isfinite(rmax) and rmax > 0:
                ax.set_rlim(0.0, 1.05 * rmax)

            # Update scatter only if something changed (prevents unnecessary redraw)
            if theta.size > 0:
                offsets = np.c_[theta, r]
                if last_offsets is None or offsets.shape != last_offsets.shape or not np.allclose(offsets, last_offsets):
                    sc.set_offsets(offsets)
                    last_offsets = offsets
                    fig.canvas.draw_idle()

            # Optional: show “status” even during gaps (plot still updates title)
            if len(df) != last_rows:
                last_rows = len(df)
                t_last = df["t"].iloc[-1]
                hc_last = int(df["has_contact"].iloc[-1]) if "has_contact" in df.columns and pd.notna(df["has_contact"].iloc[-1]) else -1
                ax.set_title(f"Contact polar map (t={t_last:.3f}, has_contact={hc_last})")

    except Exception as e:
        # If you want visibility, uncomment:
        # print("live plot error:", repr(e))
        pass

    # CRITICAL: always let Qt process events, even if no data / gap / parse issue
    fig.canvas.flush_events()
    plt.pause(0.001)
    time.sleep(0.1)
