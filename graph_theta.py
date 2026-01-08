import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv("/home/jack/sofascenes/contact_angle5.csv")

t = df["t"].to_numpy(float)
theta_unwrapped_deg = np.degrees(df["theta_unwrapped"].to_numpy(float))

fig, ax = plt.subplots()
ax.plot(t, theta_unwrapped_deg)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Theta (deg, unwrapped)")
ax.set_title("Contact circumferential angle vs time")

fmt = ScalarFormatter(useOffset=False)
fmt.set_scientific(False)
ax.yaxis.set_major_formatter(fmt)

plt.show()


# # Wrapped angle (0–360)
# plt.figure()
# plt.plot(t, theta_wrapped_deg)
# plt.xlabel("Time (s)")
# plt.ylabel("Theta (deg, wrapped)")
# plt.title("Wrapped theta vs time")
# plt.show()

# # Gap
# plt.figure()
# plt.plot(t, gap)
# plt.xlabel("Time (s)")
# plt.ylabel("Gap (m)")
# plt.title("Gap vs time")
# plt.show()
