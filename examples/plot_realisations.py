import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import ex_utils

def plot_realisations(
    mc_estimates, std_estimated, analytic_val=None, analytic_text=None
):
    """
    Violin plot of estimated quantity from Monte Carlo (MC) simulations,
    compared with error bar from estimated standard deviation.

    Also plot analytic value if specified.

    Args:
        - mc_estimates:
            1D array of quanties estimate many times by MC simulation.
        - std_estimate:
            Standard deviation estimate to be compared with standard deviation
            from MC simulations.
        - analytic_val:
            Plot horizonal line if analytic value of quantity estimated is
            provided.
        - analytic_text:
            Text to include next to line specifying analytic value, if provided.

    Returns:
        - ax:
            Plot axis.
    """

    mean = np.mean(mc_estimates)
    std_measured = np.std(mc_estimates)

    plot_aspect_ratio = 1.33
    plot_x_size = 9

    fig, ax = plt.subplots(figsize=(plot_x_size, plot_x_size / plot_aspect_ratio))

    ax.violinplot(
        mc_estimates,
        showmeans=False,
        showmedians=False,
        showextrema=True,
        bw_method=1.0,
    )

    if analytic_val is not None:
        plt.plot(np.arange(4), np.zeros(4) + analytic_val, "r--")
        ymin, ymax = ax.get_ylim()
        ax.text(1.8, analytic_val + (ymax - ymin) * 0.03, analytic_text, color="red")

    plt.errorbar(
        np.zeros(1) + 1.0,
        mean,
        yerr=std_measured,
        fmt="--o",
        color="C4",
        capsize=7,
        capthick=3,
        linewidth=3,
        elinewidth=3,
    )
    plt.errorbar(
        np.zeros(1) + 1.5,
        mean,
        yerr=std_estimated,
        fmt="--o",
        color="C2",
        capsize=7,
        capthick=3,
        linewidth=3,
        elinewidth=3,
    )

    ax.get_xaxis().set_tick_params(direction="out")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks([1.0, 1.5])
    ax.set_xticklabels(["Measured", "Estimated"])

    ax.set_xlim([0.5, 2.0])

    return ax


def remove_outliers_iqr(data, multiplier=1.5):
    """Remove outliers using IQR method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    mask = (data >= lower_bound) & (data <= upper_bound)
    return mask


savefigs = True

# Parse arguments.
parser = argparse.ArgumentParser(
    "Create violin plot of inverse evidences" + "from many realisations"
)
parser.add_argument(
    "filename_realisations",
    metavar="filename_realisations",
    type=str,
    help="Name of file containing realisations",
)
parser.add_argument(
    "filename_analytic",
    metavar="filename_analytic",
    type=str,
    help="Name of file containing analytic inverse variance",
)
parser.add_argument(
    "--remove-outliers",
    action="store_true",
    help="Remove outliers using IQR method",
)
parser.add_argument(
    "--iqr-multiplier",
    type=float,
    default=1.5,
    help="IQR multiplier for outlier detection (default: 1.5)",
)
args = parser.parse_args()

# Load data.
evidence_inv_summary = np.loadtxt(args.filename_realisations)
evidence_inv_realisations = evidence_inv_summary[:, 0]
evidence_inv_std_neg_realisations = evidence_inv_summary[:, 1]
evidence_inv_std_pos_realisations = evidence_inv_summary[:, 2]
evidence_inv_var_realisations = evidence_inv_summary[:, 3]
evidence_inv_var_var_realisations = evidence_inv_summary[:, 4]

# Remove outliers if requested
outlier_suffix = ""
outlier_realisations = None
outlier_indices = None
if args.remove_outliers:
    print(f"Original data size: {len(evidence_inv_realisations)}")
    mask = remove_outliers_iqr(evidence_inv_realisations, args.iqr_multiplier)
    
    # Calculate statistics before filtering
    mean_all = np.mean(evidence_inv_realisations)
    std_all = np.std(evidence_inv_realisations)
    
    # Store outlier information
    outlier_indices = np.where(~mask)[0]
    outlier_realisations = evidence_inv_realisations[~mask]
    outlier_std_neg = evidence_inv_std_neg_realisations[~mask]
    outlier_std_pos = evidence_inv_std_pos_realisations[~mask]
    
    # Calculate how many std deviations away each outlier is
    outlier_n_sigma = np.abs(outlier_realisations - mean_all) / std_all
    
    # Filter data
    evidence_inv_realisations = evidence_inv_realisations[mask]
    evidence_inv_std_neg_realisations = evidence_inv_std_neg_realisations[mask]
    evidence_inv_std_pos_realisations = evidence_inv_std_pos_realisations[mask]
    evidence_inv_var_realisations = evidence_inv_var_realisations[mask]
    evidence_inv_var_var_realisations = evidence_inv_var_var_realisations[mask]
    
    print(f"After removing outliers: {len(evidence_inv_realisations)}")
    print(f"Removed {np.sum(~mask)} outliers")
    
    # Print outlier information
    if len(outlier_realisations) > 0:
        print("\nOutlier values:")
        for i, (idx, val, neg, pos, n_sig) in enumerate(zip(outlier_indices, outlier_realisations, 
                                                              outlier_std_neg, outlier_std_pos, outlier_n_sigma)):
            print(f"  Index {idx}: {val:.6f} (std: {neg:.6f} / +{pos:.6f}) [{n_sig:.2f}σ away]")
    
    outlier_suffix = "_no_outliers"


print("Mean estimate:", np.mean(evidence_inv_realisations), "+-", np.std(evidence_inv_realisations))

evidence_inv_analytic = np.loadtxt(args.filename_analytic)
print("Analytic:", evidence_inv_analytic)

# Plot inverse evidence.
plt.rcParams.update({"font.size": 20})
ax = plot_realisations(
    mc_estimates=evidence_inv_realisations,
    #std_estimated=np.sqrt(np.mean(evidence_inv_var_realisations)),
    std_estimated = [[np.mean(-evidence_inv_std_neg_realisations)], [np.mean(evidence_inv_std_pos_realisations)]],
    #std_estimated = [[-evidence_inv_std_neg_realisations[1]], [evidence_inv_std_pos_realisations[1]]],
    analytic_val=evidence_inv_analytic,
    analytic_text=r"Truth",
)
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.set_ylabel(r"Log inverse evidence ($\ln \rho$)")

filename_base = os.path.basename(args.filename_realisations)
filename_base_noext = os.path.splitext(filename_base)[0]
if savefigs:
    plt.savefig(
        "./examples/plots/" + filename_base_noext + outlier_suffix + "_evidence_inv.png",
        bbox_inches="tight",
    )

# Plot variance of inverse evidence.
ax = plot_realisations(
    mc_estimates=evidence_inv_var_realisations,
    std_estimated=np.exp(evidence_inv_var_var_realisations[0])
)
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.set_ylabel(r"Inverse evidence variance ($\sigma^2$)")

if savefigs:
    plt.savefig(
        "./examples/plots/" + filename_base_noext + outlier_suffix + "_evidence_inv_var.png",
        bbox_inches="tight",
    )

plt.show(block=False)

input("\nPress Enter to continue...")