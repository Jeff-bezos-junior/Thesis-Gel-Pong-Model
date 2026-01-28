import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.stats import mannwhitneyu

# ====== USER SETTINGS ======
FOLDER_PATH = r"paste_the_path"
TOTAL_TIME = 100          # seconds (your run length)
DT = 1                    # resolution (1 second)
MEAN_SMOOTH = 5           # for short 100s experiments
SEM_SMOOTH  = 10          # keeps cloud consistent
# ===========================


# ====== LOAD FILES ======
files = [os.path.join(FOLDER_PATH, f)
         for f in os.listdir(FOLDER_PATH)
         if f.startswith("ver") and f.endswith(".csv")]

if len(files) == 0:
    print("❌ No CSV files found")
    exit()

runs = []

for f in files:
    df = pd.read_csv(f)

    # Each run is converted into a 1-second uniform series
    t = df["time"].values
    rl = df["rally_length"].values

    # step function: use last known rally length
    ts = np.zeros(TOTAL_TIME)

    last = 0
    idx = 0

    for sec in range(TOTAL_TIME):
        # advance through events
        while idx < len(t) and t[idx] <= sec:
            last = rl[idx]
            idx += 1

        ts[sec] = last

    runs.append(ts)

runs = np.array(runs)   # shape: (num_runs, TOTAL_TIME)


# ====== COMPUTE MEAN + SEM ======
mean_curve = np.mean(runs, axis=0)
sem_curve  = np.std(runs, axis=0) / np.sqrt(runs.shape[0])

# ====== SMOOTH ======
smooth_mean = uniform_filter1d(mean_curve, size=MEAN_SMOOTH)
smooth_sem  = uniform_filter1d(sem_curve,  size=SEM_SMOOTH)

# ====== P-VALUE ======
mid = TOTAL_TIME // 2
pre  = smooth_mean[:mid]
post = smooth_mean[mid:]
_, p_value = mannwhitneyu(pre, post, alternative="less")

# ====== PLOT ======
plt.figure(figsize=(12,6))
x = np.arange(TOTAL_TIME)

# SEM cloud
plt.fill_between(x,
                 smooth_mean - smooth_sem,
                 smooth_mean + smooth_sem,
                 color="gold",
                 alpha=0.25,
                 label="SEM")

# Smooth mean line
plt.plot(x, smooth_mean, color="goldenrod", linewidth=3, label="Mean (smoothed)")

# Raw mean
plt.plot(x, mean_curve, color="gray", alpha=0.3, linewidth=1, label="Raw mean")

# p-value display
plt.text(0.02, 0.92,
         f"p = {p_value:.15f}",
         transform=plt.gca().transAxes,
         fontsize=12,
         fontweight="bold",
         bbox=dict(facecolor="white", edgecolor="black", alpha=0.8))

plt.title("Aggregate Rally Length (Mean ± SEM)")
plt.xlabel("Time (s)")
plt.ylabel("Rally Length")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
