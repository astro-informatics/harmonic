import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rcParams['axes.formatter.useoffset'] = False

name = "RQSpline_rosenbrock_evidence_v_T_realisations"
#name = "examples/plots/RQSpline_10D_200ch_10000000s_5000bi_3l_32b_30e__rosenbrock_ln_evidence_inv_vary_T"
name = "examples/plots/RQSpline_10D_200ch_10000000s_5000bi_3l_32b_30e_rosenbrock_ln_evidence_inv_vary_T"
results = np.loadtxt(name + ".dat")
#analytic = np.loadtxt("RQSpline_rosenbrock_inv_of_log_evidence_analytic.dat")
#print("Analytic ", analytic)
results_df = pd.DataFrame(
    results,
    columns=[
        "Ln evidence inv",
        "Negative errors",
        "Positive errors",
        "Ln evidence inv var",
        "Ln evidence inv var var",
        "Temperature"
    ],
)

ln_evidence_inv_ref = [43.145, 43.108]
error_ref = [0.023, 0.551]
labels = ["nautilus", "dynesty-s"]
colors = ["r", "blue"]

ln_evidence_inv_ref = [43.145]
error_ref = [0.023]
labels = ["nautilus"]
colors = ["r"]


plt.errorbar(results_df["Temperature"], results_df["Ln evidence inv"], yerr= [-results_df["Negative errors"], results_df["Positive errors"]], capsize=3, fmt=".", ecolor = "black")
if False:
    for i in range(len(ln_evidence_inv_ref)):
        plt.axhline(ln_evidence_inv_ref[i], color = colors[i], label=labels[i])
        plt.axhline(ln_evidence_inv_ref[i] - error_ref[i], color = colors[i], label=labels[i], linestyle="--")
        plt.axhline(ln_evidence_inv_ref[i] + error_ref[i], color = colors[i], label=labels[i], linestyle="--")
plt.xlabel("Temperature")
plt.ylabel("Log inverse evidence for 10D Rosenbrock")
#plt.legend()
plt.savefig(name + ".png", dpi=300)
plt.show()


#plot with reference values

# Example data
temperatures = results_df["Temperature"]  # Temperature values
evidence_values = results_df["Ln evidence inv"]
asym_errors_low = -results_df["Negative errors"]
asym_errors_high = results_df["Positive errors"]

# Competing methods
methods = {
    "harmonic (T=0.8)": (results_df.query('Temperature == 0.8')["Ln evidence inv"], [-results_df.query('Temperature == 0.8')["Negative errors"], results_df.query('Temperature == 0.8')["Positive errors"]]),
    "nautilus*": (43.145, 0.023),
    "nautilus-r*": (43.176, 0.024),
    "dynesty-r*": (43.175, 0.747),
    "dynesty-s*": (43.108, 0.551),
    "pocoMC*": (43.745, 0.222),
    "UltraNest-m*": (43.290, 0.317),
}
colors = ["black", "red", "darkred", "blue", "navy", "green", "blueviolet"]

reference_temperatures = np.linspace(min(temperatures), max(temperatures), num = len(methods))
xticks = []
# Plot setup
#fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
#fig.subplots_adjust(hspace=0.3)

# Subplot 1: Our method + competing methods
#ax1 = axes[0]
fig, ax1 = plt.subplots(figsize=(10, 4))
if False:
    ax1.errorbar(
    temperatures, evidence_values, yerr=[asym_errors_low, asym_errors_high],
    fmt='o', label="Learned harmonic mean", color="black", capsize=4)
    for (method, (value, error)), color in zip(methods.items(), colors):
        ax1.plot(temperatures, [value] * len(temperatures), color=color, label=method)
        ax1.plot(temperatures, [value - error] * len(temperatures), color=color, linestyle="--")
        ax1.plot(temperatures, [value + error] * len(temperatures), color=color, linestyle="--")

for temp, (method, (value, error)), color in zip(reference_temperatures, methods.items(), colors):
    ax1.errorbar(
        temp, value, yerr=error, fmt='o', capsize=6, color=color
    )
    xticks.append(method)
# inset Axes....
x1, x2, y1, y2 = 0.495, 0.605, 43.1, 43.205  # subregion of the original image
axins = ax1.inset_axes(
    [0.08, 0.62, 0.4, 0.35],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
for temp, (method, (value, error)), color in zip(reference_temperatures, methods.items(), colors):
    axins.errorbar(
        temp, value, yerr=error, fmt='.', capsize=6, color=color, markersize=10
    )
ax1.indicate_inset_zoom(axins, edgecolor="black")
#ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)  # Remove x-axis ticks
ax1.set_xticks(reference_temperatures)
ax1.set_xticklabels(xticks, fontsize=11.5)
ax1.set_ylabel("Log reciprocal evidence", fontsize=13)
#ax1.set_title("Log reciprocal evidence for 10D Rosenbrock")
#ax1.legend(loc = "upper left", fontsize="small")
ax1.grid()
axins.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.savefig(name + "_ref.png", dpi=300)
plt.show()

# Subplot 2: Zoomed-in view of our method
#ax2 = axes[1]
fig, ax2 = plt.subplots(figsize=(10, 4))
ax2.errorbar(
    temperatures, evidence_values, yerr=[asym_errors_low, asym_errors_high],
    fmt='o', color="black", capsize=6
)
ax2.set_ylabel("Log reciprocal evidence", fontsize=13)
#ax2.set_title("Learned harmonic mean log reciprocal evidence for varying temperature")
#ax2.set_ylim([0, max(evidence_values) + 0.1 * max(evidence_values)])
ax2.set_xlabel("Temperature", fontsize=13)
ax2.grid()
fig.subplots_adjust(bottom=0.2)
plt.savefig(name + ".png", dpi=300)
plt.show()
