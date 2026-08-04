import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# x-axis (percentages)
p = np.array([100, 75, 50, 25, 10, 1])
xticklabels = [f"{x}%" for x in p]

# ============================================================
# TOP PLOT — Open-loop
# ============================================================

T1 = {
    "DIAYN":       (np.array([8.779, 9.171, 8.952, 8.486, 8.861, 10.914]),
                    np.array([0.340, 0.362, 0.345, 0.351, 0.356, 0.362])),
    "Disagreement":(np.array([7.587, 7.374, 7.505, 7.166, 7.781, 10.071]),
                    np.array([0.317, 0.312, 0.312, 0.315, 0.326, 0.361])),
    "ICM":         (np.array([8.333, 7.724, 7.631, 7.341, 8.231, 10.956]),
                    np.array([0.337, 0.317, 0.318, 0.302, 0.330, 0.365])),
    "ICM-APT":     (np.array([9.256, 8.764, 9.174, 8.513, 9.403, 12.075]),
                    np.array([0.359, 0.360, 0.359, 0.365, 0.373, 0.358])),
    "PROTO":       (np.array([7.669, 7.225, 7.867, 7.556, 8.120, 10.841]),
                    np.array([0.321, 0.313, 0.345, 0.341, 0.337, 0.361])),
    "Random":      (np.array([9.003, 8.371, 8.440, 8.607, 8.705, 10.713]),
                    np.array([0.364, 0.349, 0.342, 0.344, 0.357, 0.365])),
    "RND":         (np.array([9.111, 8.666, 9.209, 8.518, 9.398, 11.364]),
                    np.array([0.367, 0.351, 0.366, 0.354, 0.358, 0.347])),
    "SMM":         (np.array([9.008, 8.717, 9.198, 9.085, 9.584, 12.690]),
                    np.array([0.358, 0.361, 0.360, 0.367, 0.360, 0.363])),
    "Scratch":     (np.array([9.669, 9.369, 9.990, 9.968, 10.842, 14.371]),
                    np.array([0.365, 0.356, 0.338, 0.352, 0.386, 0.391])),
}

import seaborn as sns
colors = sns.color_palette("nipy_spectral", n_colors=len(T1))


# ============================================================
# BOTTOM PLOT — Closed-loop
# ============================================================

T2 = {
    "DIAYN":       (np.array([5.779, 5.573, 5.688, 5.683, 5.709, 6.904]),
                    np.array([0.325, 0.305, 0.297, 0.311, 0.307, 0.306])),
    "Disagreement":(np.array([5.741, 5.523, 5.718, 6.054, 5.846, 6.946]),
                    np.array([0.317, 0.312, 0.318, 0.341, 0.317, 0.301])),
    "ICM":         (np.array([5.613, 5.673, 5.708, 5.812, 5.818, 7.525]),
                    np.array([0.318, 0.324, 0.319, 0.326, 0.327, 0.338])),
    "ICM-APT":     (np.array([6.732, 6.533, 6.591, 6.874, 6.770, 8.008]),
                    np.array([0.343, 0.342, 0.358, 0.360, 0.354, 0.323])),
    "PROTO":       (np.array([5.005, 4.994, 5.168, 5.071, 5.488, 7.415]),
                    np.array([0.269, 0.275, 0.278, 0.280, 0.307, 0.337])),
    "Random":      (np.array([5.229, 5.162, 5.489, 5.631, 5.688, 6.588]),
                    np.array([0.304, 0.290, 0.298, 0.311, 0.325, 0.313])),
    "RND":         (np.array([6.405, 6.114, 6.285, 6.142, 6.316, 8.184]),
                    np.array([0.335, 0.320, 0.346, 0.327, 0.354, 0.330])),
    "SMM":         (np.array([6.682, 6.471, 6.759, 6.645, 6.921, 8.302]),
                    np.array([0.332, 0.316, 0.336, 0.340, 0.350, 0.301])),
    "Scratch":     (np.array([8.545, 7.872, 7.235, 8.638, 7.887, 16.294]),
                    np.array([0.348, 0.310, 0.311, 0.364, 0.320, 0.406])),
}

# Plots --------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

# --- TOP PLOT ---
for (label, (y, yerr)), color in zip(T1.items(), colors):
    axs[0].errorbar(p, y, yerr=yerr, marker="o", label=label, color=color)

axs[0].set_title("Open Loop")
axs[0].grid(True)
axs[0].legend()
axs[0].set_xticks(p)
axs[0].set_xticklabels(xticklabels)

# --- BOTTOM PLOT ---
for (label, (y, yerr)), color in zip(T2.items(), colors):
    axs[1].errorbar(p, y, yerr=yerr, marker="o", label=label, color=color)


axs[1].set_title("Closed Loop")
axs[1].grid(True)
axs[1].legend()
axs[1].set_xticks(p)
axs[1].set_xticklabels(xticklabels)

axs[0].legend(loc="upper left", ncol=2)
axs[1].legend(loc="upper left", ncol=2)

plt.xlabel("Dataset Split")
fig.text(0.005, 0.5, "Distance to Goal", va='center', rotation='vertical')

plt.get_current_fig_manager().set_window_title("Walker_run task Dist2Goal for different exploratory pretraining algorithms.")

# plt.title("Walker_run task Dist2Goal for different exploratory pretraining algorithms.")

plt.gca().invert_xaxis()
# plt.grid(True)

plt.tight_layout()
plt.get_current_fig_manager().window.wm_geometry("+50+30")
plt.show()
