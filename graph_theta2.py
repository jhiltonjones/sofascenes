import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/home/jack/sofascenes/tip_contacts_all.csv")

plt.figure()
sc = plt.scatter(df["t"], np.rad2deg(df["theta"]), c=df["gap_surf"], s=8)
plt.xlabel("t (s)")
plt.ylabel("theta (deg)")
plt.title("All tip-local contacts: theta vs time (colored by surface gap)")
plt.colorbar(sc, label="gap_surf (mm)")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/home/jack/sofascenes/tip_contacts_all.csv")

theta = df["theta"].to_numpy(dtype=float)     # radians
r = df["t"].to_numpy(dtype=float)             # time as radius
c = df["gap_surf"].to_numpy(dtype=float)      # color

fig = plt.figure()
ax = fig.add_subplot(111, projection="polar")
sc = ax.scatter(theta, r, c=c, s=8)

ax.set_title("Tip-local contacts (polar): angle=theta, radius=time")
ax.set_xlabel("theta (rad)")   # polar plots don’t really use x/y labels, but ok
ax.set_ylabel("t (s)")

cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label("gap_surf (mm)")

plt.show()
