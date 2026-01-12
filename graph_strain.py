import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

spin = np.genfromtxt("/home/jack/sofascenes/tip_spin.csv", delimiter=",", names=True)
curv = np.genfromtxt("/home/jack/sofascenes/curv_tau.csv", delimiter=",", names=True)

plt.figure()
plt.plot(spin["t"], np.rad2deg(spin["phi_unwrapped"]))
plt.xlabel("t (s)")
plt.ylabel("tip spin (deg)")
plt.title("Tip spin vs time")
plt.show()


t0 = float(spin["t"][-1])
idx = np.argmin(np.abs(curv["t"] - t0))
t0_data = float(curv["t"][idx])

mask_t = (curv["t"] == t0_data)
slice_s = curv["s"][mask_t]
slice_tau = curv["tau"][mask_t]
slice_kappa = curv["kappa"][mask_t]

s_min = 0 
mask_s = (slice_s >= s_min) & np.isfinite(slice_s)
slice_s = slice_s[mask_s]
slice_tau = slice_tau[mask_s]
slice_kappa = slice_kappa[mask_s]

order = np.argsort(slice_s)


# plt.figure()
# plt.plot(slice_s[order], slice_tau[order])  
# plt.xlabel("s (mm)")
# plt.ylabel("tau (rad/mm)")
# plt.title(f"tau(s) for s≥{s_min} mm at t={t0_data:.3f} s")
# plt.show()


plt.figure()
plt.plot(slice_s[order], slice_kappa[order]) 
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


df = pd.read_csv("/home/jack/sofascenes/curv_tau.csv")
t0 = df["t"].iloc[-1]
sl = df[df["t"] == t0].sort_values("s")

s = sl["s"].to_numpy(float)      
k = sl["kappa"].to_numpy(float)   

mask = np.isfinite(s) & np.isfinite(k)
s = s[mask]; k = k[mask]

theta_total = np.trapz(k, s)
print("Total bend angle (rad):", theta_total)
print("Total bend angle (deg):", np.rad2deg(theta_total))
