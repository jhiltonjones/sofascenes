import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv("/home/jack/sofascenes/contact_angle5.csv")

t = df["t"].to_numpy(float)
theta_unwrapped_deg = np.degrees(df["theta_unwrapped"].to_numpy(float))
theta_wrapped_deg = np.degrees(df["theta_wrapped"].to_numpy(float))

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
plt.figure()
plt.plot(t, theta_wrapped_deg)
plt.xlabel("Time (s)")
plt.ylabel("Theta (deg, wrapped)")
plt.title("Wrapped theta vs time")
plt.show()

# # Gap
# plt.figure()
# plt.plot(t, gap)
# plt.xlabel("Time (s)")
# plt.ylabel("Gap (m)")
# plt.title("Gap vs time")
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/home/jack/sofascenes/contact_angle5.csv")

# If you have has_contact, use it. Otherwise use NaN theta as proxy.
if "has_contact" in df.columns:
    mask = df["has_contact"].to_numpy(int) == 1
else:
    mask = np.isfinite(df["theta_wrapped"].to_numpy(float))

t = df.loc[mask, "t"].to_numpy(float)
theta = df.loc[mask, "theta_wrapped"].to_numpy(float)  # radians already

fig = plt.figure()
ax = fig.add_subplot(111, projection="polar")
ax.scatter(theta, t, s=5)  # theta is angle, t is radius
ax.set_title("Contact angle around circumference (radius = time)")
ax.set_rlabel_position(135)
plt.show()

