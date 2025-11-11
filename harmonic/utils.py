from typing import Tuple, List
import numpy as np
import harmonic as hm
import getdist
import matplotlib as plt
import getdist.plots


def eval_func_on_grid(func, xmin, xmax, ymin, ymax, nx, ny):
    """
    Evalute 2D function on a grid.

    Args:
        - func:
            Function to evalate.
        - xmin:
            Minimum x value to consider in grid domain.
        - xmax:
            Maximum x value to consider in grid domain.
        - ymin:
            Minimum y value to consider in grid domain.
        - ymax:
            Maximum y value to consider in grid domain.
        - nx:
            Number of samples to include in grid in x direction.
        - ny:
            Number of samples to include in grid in y direction.

    Returns:
        - func_eval_grid:
            Function values evaluated on the 2D grid.
        - x_grid:
            x values over the 2D grid.
        - y_grid:
            y values over the 2D grid.
    """

    # Evaluate func over grid.
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    x_grid, y_grid = np.meshgrid(x, y)
    func_eval_grid = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            func_eval_grid[i, j] = func(np.array([x_grid[i, j], y_grid[i, j]]))

    return func_eval_grid, x_grid, y_grid


def plot_getdist(samples, labels=None):
    """
    Plot triangle plot of marginalised distributions using getdist package.

    Args:
        - samples:
            2D array of shape (nsamples, ndim) containing samples.
        - labels:
            Array of strings containing axis labels.

    Returns:
        - None
    """

    getdist.chains.print_load_details = False

    ndim = samples.shape[1]
    names = ["x%s" % i for i in range(ndim)]
    if labels is None:
        labels = ["x_%s" % i for i in range(ndim)]

    mcsamples = getdist.MCSamples(samples=samples, names=names, labels=labels)
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot([mcsamples], filled=True)


def plot_getdist_compare(
    samples1, samples2, labels=None, fontsize=17, legend_fontsize=15
):
    """
    Plot triangle plot of marginalised distributions using getdist package.

    Args:
        samples1 : 2D array of shape (nsamples, ndim) containing samples from the posterior.

        samples2 : 2D array of shape (nsamples, ndim) containing samples from the concentrated flow.

        labels: Array of strings containing axis labels for both sets of samples.

        fontsize: Plot fontsize.

        legend_fontsize: Plot legend fontsize.

    Returns:
        None
    """

    getdist.chains.print_load_details = False
    # getdist.plots.GetDistPlotSettings(constrained_layout=True)

    ndim = samples1.shape[1]
    names = ["x%s" % i for i in range(ndim)]
    if labels is None:
        labels = ["x_%s" % i for i in range(ndim)]

    mcsamples1 = getdist.MCSamples(
        samples=samples1, names=names, labels=labels, label="Posterior"
    )

    mcsamples2 = getdist.MCSamples(
        samples=samples2, names=names, labels=labels, label="Concentrated flow"
    )

    g = getdist.plots.getSubplotPlotter(width_inch=10.5 / 2.54)
    # g.settings.scaling = False
    g.settings.axes_fontsize = fontsize
    g.settings.legend_fontsize = legend_fontsize  # 17.5
    g.settings.axes_labelsize = fontsize
    g.settings.linewidth = 2
    g.settings.constrained_layout = True
    g.settings.legend_loc = "upper right"
    g.triangle_plot(
        [mcsamples1, mcsamples2],
        filled=True,
        contour_colors=["red", "tab:blue"],
        line_args=[{"ls": "-", "color": "red"}, {"ls": "--", "color": "blue"}],
    )


def split_data(chains, training_proportion: float = 0.5) -> Tuple:
    """Split the data in a chains instance into two (e.g. training and test sets).

    New chains instances can be used for training and calculation the evidence
    on the "test" set.

    Chains are split so that the first chains in the original chains object go
    into the training set and the following go into the test set.

    Args:

        chains (Chains): Instance of a chains class containing the data to be split.

        training_proportion (float): Proportion of data to be used in training
            (default=0.5)

    Returns:

        (Chains, Chains): A tuple containing the following two Chains.

            chains_train (Chains): Instance of a chains class containing
              chains to be used to fit the model (e.g. training).

            chains_test (Chains): Instance of a chains class containing
              chains to be used to calculate the evidence (e.g. testing).

    Raises:

        ValueError: Raised if training_proportion is not strictly between 0 and
            1.

        ValueError: Raised if resulting nchains in training set is less than 1.

        ValueError: Raised if resulting nchains in test set is less than 1.

    """

    if training_proportion <= 0.0 or training_proportion >= 1.0:
        raise ValueError("training_proportion must be strictly between " "0 and 1.")

    nchains_train = int(chains.nchains * training_proportion)
    nchains_test = chains.nchains - nchains_train

    if nchains_train < 1:
        raise ValueError("nchains for training set must strictly greater " "than 0.")
    if nchains_test < 1:
        raise ValueError("nchains for test set must strictly greater than 0.")

    ndim = chains.ndim

    chains_train = hm.chains.Chains(ndim)
    chains_test = hm.chains.Chains(ndim)

    start_index = chains.start_indices[0]
    end_index = chains.start_indices[nchains_train]
    chains_train.add_chains_2d_list(
        chains.samples[start_index:end_index, :],
        chains.ln_posterior[start_index:end_index],
        nchains_train,
        chains.start_indices[: nchains_train + 1],
    )

    start_index = chains.start_indices[nchains_train]
    end_index = chains.start_indices[-1]
    chains_test.add_chains_2d_list(
        chains.samples[start_index:end_index, :],
        chains.ln_posterior[start_index:end_index],
        nchains_test,
        chains.start_indices[nchains_train:],
    )

    return chains_train, chains_test


def validation_fit_indexes(
    i_fold: int, nchains_in_val_set: int, nfold: int, indexes
) -> Tuple[List, List]:
    """Extract the correct indexes for the chains of the validation and training
    sets.

    Args:

        i_fold (int): Cross-validation iteration to perform.

        nchains_in_val_set (int): The number of chains that will go in each
            validation set.

        nfold (int): Number of fold validation sets to be made.

        indexes (List): List of the chains to be used in fold validation that
            need to be split.

    Returns:

        (List, List): A tuple containing the following two lists of indices.

            indexes_val (List): List of indexes for the validation set.

            indexes_fit (List): List of indexes for the training set.

    Raises:

        ValueError: Raised if the value of i_fold does not fall between 0 and
            nfold-1.

    """

    if i_fold < 0 or i_fold >= nfold:
        raise ValueError("i_fold is not the range set by nfold")

    if nchains_in_val_set < 1 or nchains_in_val_set >= len(indexes):
        raise ValueError(
            "nchains_in_val_set must be strictly between 0 " "and length of indexes."
        )

    if i_fold < nfold - 1:
        indexes_val = indexes[
            i_fold * nchains_in_val_set : (i_fold + 1) * nchains_in_val_set
        ]
        indexes_fit = (
            indexes[: i_fold * nchains_in_val_set]
            + indexes[(i_fold + 1) * nchains_in_val_set :]
        )
    else:
        indexes_val = indexes[(i_fold) * nchains_in_val_set :]
        # ensures all the chains get used even if nchains % nfold != 0
        indexes_fit = indexes[: i_fold * nchains_in_val_set]

    return indexes_val, indexes_fit


def cross_validation(
    chains,
    domains: List,
    hyper_parameters: List,
    nfold=2,
    modelClass=None,
    seed: int = -1,
) -> List:
    """Perform n-fold validation for given model using chains to be split into
    validation and training data.

    First, splits data into nfold chunks. Second, fits the model using each of
    the hyper-parameters given using all but one of the chunks (the validation
    chunk). This procedure is performed for all the chunks and the average
    (mean) log-space variance from all the chunks is computed and returned.
    This can be used to decide which hyper-parameters list was better.

    Args:

        chains (Chains): Chains containing samples (to be split into
            training and validation data herein).

        domains (List): Domains of the model's parameters.

        hyper_parameters (List): List of hyper_parameters where each entry is a
            hyper_parameter list to be considered.

        modelClass (Model): Model that is being cross validated (defaults to
            KernelDensityEstimate inside function).

        seed (int): Seed for random number generator when drawing the chains
            (if this is negative the seed is not set).

    Returns:

        (List): Mean log validation variance (averaged over nfolds) for each hyper-parameter.

    Raises:

        ValueError: Raised if model is not one of the posible models.

    """

    if modelClass is None:
        modelClass = hm.model_legacy.KernelDensityEstimate

    ln_validation_variances = np.zeros((nfold, len(hyper_parameters)))

    if seed > 0:
        np.random.seed(seed)

    indexes = list(np.random.permutation(chains.nchains))

    nchains_in_val_set = int(chains.nchains / nfold)

    for i_fold in range(nfold):
        indexes_val, indexes_fit = validation_fit_indexes(
            i_fold, nchains_in_val_set, nfold, indexes
        )

        chains_val = chains.get_sub_chains(indexes_val)
        chains_fit = chains.get_sub_chains(indexes_fit)

        for i_val, hyper_parameter in enumerate(hyper_parameters):
            model = modelClass(chains.ndim, domains, hyper_parameters=hyper_parameter)

            # Fit model
            model.fit(chains_fit.samples, chains_fit.ln_posterior)

            # Calculate evidence
            ev = hm.evidence.Evidence(chains_val.nchains, model)
            ev.add_chains(chains_val)

            hm.logs.debug_log(
                "cross_validation: ifold = {}; hyper_parameter = {}".format(
                    i_fold, hyper_parameter
                )
            )
            hm.logs.debug_log(
                "cross_validation: evidence_inv = {}".format(ev.evidence_inv)
            )
            hm.logs.debug_log(
                "cross_validation: evidence_inv_var = {}".format(ev.evidence_inv_var)
            )
            hm.logs.debug_log(
                "cross_validation:"
                + " evidence_inv_var**0.5/evidence_inv = {}".format(
                    ev.evidence_inv_var**0.5 / ev.evidence_inv
                )
            )
            hm.logs.debug_log(
                "cross_validation: evidence_inv_var_var = {}".format(
                    ev.evidence_inv_var_var
                )
            )

            ln_validation_variances[i_fold, i_val] = ev.ln_evidence_inv_var

    return np.nanmean(ln_validation_variances, axis=0)
