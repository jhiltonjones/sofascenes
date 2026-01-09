import numpy as np
import matplotlib.pyplot as plt

spin = np.genfromtxt("/home/jack/sofascenes/tip_spin.csv", delimiter=",", names=True)
curv = np.genfromtxt("/home/jack/sofascenes/curv_tau.csv", delimiter=",", names=True)

plt.figure()
plt.plot(spin["t"], np.rad2deg(spin["phi_unwrapped"]))
plt.xlabel("t (s)")
plt.ylabel("tip spin (deg)")
plt.show()

t0 = float(spin["t"][-1])
idx = np.argmin(np.abs(curv["t"] - t0))
t0_data = float(curv["t"][idx])

mask_t = (curv["t"] == t0_data)
slice_s = curv["s"][mask_t]
slice_tau = curv["tau"][mask_t]

# drop first 10 mm of arclength
mask_s = (slice_s >= 20.0)
slice_s = slice_s[mask_s]
slice_tau = slice_tau[mask_s]

order = np.argsort(slice_s)

plt.figure()
plt.plot(slice_s[order], slice_tau[order])
plt.xlabel("s (mm)")
plt.ylabel("tau (rad/mm)")
plt.title(f"tau(s) for s≥10 mm at t={t0_data:.3f}")
plt.show()
