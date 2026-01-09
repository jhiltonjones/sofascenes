import numpy as np
import matplotlib.pyplot as plt

spin = np.genfromtxt("/home/jack/sofascenes/tip_spin.csv", delimiter=",", names=True)
curv = np.genfromtxt("/home/jack/sofascenes/curv_tau.csv", delimiter=",", names=True)

# -------------------------
# 1) Tip spin vs time
# -------------------------
plt.figure()
plt.plot(spin["t"], np.rad2deg(spin["phi_unwrapped"]))
plt.xlabel("t (s)")
plt.ylabel("tip spin (deg)")
plt.title("Tip spin vs time")
plt.show()

# -------------------------
# Pick a time slice: nearest time in curv_tau to latest spin time
# -------------------------
t0 = float(spin["t"][-1])
idx = np.argmin(np.abs(curv["t"] - t0))
t0_data = float(curv["t"][idx])

mask_t = (curv["t"] == t0_data)
slice_s = curv["s"][mask_t]
slice_tau = curv["tau"][mask_t]
slice_kappa = curv["kappa"][mask_t]

# Optional: clip out the base / keep s >= threshold
s_min = 10  # mm (change as desired)
mask_s = (slice_s >= s_min) & np.isfinite(slice_s)
slice_s = slice_s[mask_s]
slice_tau = slice_tau[mask_s]
slice_kappa = slice_kappa[mask_s]

order = np.argsort(slice_s)

# -------------------------
# 2) tau(s) profile at t0
# -------------------------
plt.figure()
plt.plot(slice_s[order], slice_tau[order])  # tau is already in rad/mm (per your code)
plt.xlabel("s (mm)")
plt.ylabel("tau (rad/mm)")
plt.title(f"tau(s) for s≥{s_min} mm at t={t0_data:.3f} s")
plt.show()

# -------------------------
# 3) kappa(s) profile at t0
# -------------------------
plt.figure()
plt.plot(slice_s[order], slice_kappa[order])  # kappa is 1/mm (given your units are mm)
plt.xlabel("s (mm)")
plt.ylabel("kappa (1/mm)")
plt.title(f"kappa(s) for s≥{s_min} mm at t={t0_data:.3f} s")
plt.show()

plt.figure()
plt.plot(slice_s[order], slice_kappa[order] * (180.0/np.pi))
plt.xlabel("s (mm)")
plt.ylabel("curvature (deg/mm)")
plt.title(f"curvature (deg/mm) for s≥{s_min} mm at t={t0_data:.3f} s")
plt.show()

