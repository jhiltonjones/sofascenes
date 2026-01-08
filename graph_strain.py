import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("/home/jack/sofascenes/curv_tau.csv")

t0 = df["t"].max()
snap = df[df["t"] == t0].sort_values("s")

snap = snap[snap["s"] >= 10.0]   # ignore first 10 along the catheter

plt.figure()
plt.plot(snap["s"].to_numpy(), snap["kappa"].to_numpy())
plt.xlabel("s (arc-length along catheter)")
plt.ylabel("curvature κ (1/length units)")
plt.title(f"Curvature profile at t={t0:.3f}s")
plt.show()

plt.figure()
plt.plot(snap["s"].to_numpy(), snap["tau"].to_numpy())
plt.xlabel("s (arc-length along catheter)")
plt.ylabel("torsion τ (1/length units)")
plt.title(f"Torsion profile at t={t0:.3f}s")
plt.show()
