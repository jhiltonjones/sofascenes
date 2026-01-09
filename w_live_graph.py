import os
import time
import numpy as np
import pandas as pd

os.environ["MPLBACKEND"] = "QtAgg"
import matplotlib.pyplot as plt

CSV = "/home/jack/sofascenes/tip_contacts_all.csv"

# Optional: only show contacts with small surface gap (mm); set to None to show all rows
GAP_MAX = None          # e.g. 0.5 for <=0.5 mm, or None for no filter

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="polar")

# initialize an empty scatter; we will set offsets + colors later
sc = ax.scatter([], [], s=8)

ax.set_title("Tip-local contacts (polar): angle=theta, radius=time")
ax.set_rlabel_position(135)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

fig.show()

last_good = None
last_offsets = None
last_colors = None
last_rows = 0

def read_csv_robust(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    except Exception:
        return None

    required = {"t", "theta", "gap_surf"}
    if not required.issubset(df.columns):
        return None

    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["theta"] = pd.to_numeric(df["theta"], errors="coerce")
    df["gap_surf"] = pd.to_numeric(df["gap_surf"], errors="coerce")

    # contact_index/id1/id2/d_tip may exist; keep them if present
    return df

while True:
    try:
        df = read_csv_robust(CSV)
        if df is not None and len(df) > 0:
            last_good = df

        if last_good is not None:
            df = last_good

            mask = np.isfinite(df["t"]) & np.isfinite(df["theta"]) & np.isfinite(df["gap_surf"])
            if GAP_MAX is not None:
                mask &= (df["gap_surf"] <= float(GAP_MAX))

            theta = df.loc[mask, "theta"].to_numpy(float)     # radians
            r = df.loc[mask, "t"].to_numpy(float)             # time
            c = df.loc[mask, "gap_surf"].to_numpy(float)      # color

            # update radial limit based on all times (or contact-only; choose preference)
            rmax = float(np.nanmax(df["t"].to_numpy(float)))
            if np.isfinite(rmax) and rmax > 0:
                ax.set_rlim(0.0, 1.05 * rmax)

            # Build polar scatter offsets: (theta, r)
            offsets = np.c_[theta, r] if theta.size else np.empty((0, 2), float)

            # Only redraw if changed
            changed = False
            if last_offsets is None or offsets.shape != last_offsets.shape:
                changed = True
            elif offsets.size and not np.allclose(offsets, last_offsets, equal_nan=True):
                changed = True

            if last_colors is None or c.shape != (last_colors.shape if last_colors is not None else (0,)):
                changed = True
            elif c.size and last_colors is not None and not np.allclose(c, last_colors, equal_nan=True):
                changed = True

            if changed:
                sc.set_offsets(offsets)

                # Set colors (gap_surf) and keep color scaling stable-ish
                if c.size:
                    sc.set_array(c)
                    # Optional: auto-scale colorbar range as data evolves
                    vmin = float(np.nanmin(c))
                    vmax = float(np.nanmax(c))
                    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                        sc.set_clim(vmin, vmax)

                last_offsets = offsets
                last_colors = c.copy() if c.size else np.array([], float)

                fig.canvas.draw_idle()

            # Update title when file grows
            if len(df) != last_rows:
                last_rows = len(df)
                t_last = float(df["t"].iloc[-1]) if pd.notna(df["t"].iloc[-1]) else np.nan
                ax.set_title(f"Tip-local contacts (polar) — last t={t_last:.3f}s, rows={last_rows}")

    except Exception:
        pass

    fig.canvas.flush_events()
    plt.pause(0.001)
    time.sleep(0.1)
