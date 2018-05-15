import numpy as np
import sys
import emcee
import time
import matplotlib.pyplot as plt
from functools import partial
sys.path.append(".")
import harmonic as hm
sys.path.append("examples")
import utils


def ln_prior(x, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):
    """Compute log_e of uniform prior.

    Args:
        x: Position at which to evaluate prior.
        xmin: Uniform prior minimum x edge (first dimension).
        xmax: Uniform prior maximum x edge (first dimension).
        ymin: Uniform prior minimum y edge (second dimension).
        ymax: Uniform prior maximum y edge (second dimension).

    Returns:
        double: Value of prior at specified point.
    """

    if x[0] >= xmin and x[0] <= xmax and x[1] >= ymin and x[1] <= ymax:
        return 1.0 / ( (xmax - xmin) * (ymax - ymin) )
    else:
        return 0.0


def ln_likelihood(x):
    """Compute log_e of likelihood defined by Himmelblau function.

    Args:
        x: Position at which to evaluate likelihood.

    Returns:
        double: Value of Himmelblau function at specified point.
    """

    f = (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2


    return -f


def ln_posterior(x, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0):
    """Compute log_e of posterior.

    Args:
        x: Position at which to evaluate posterior.
        xmin: Uniform prior minimum x edge (first dimension).
        xmax: Uniform prior maximum x edge (first dimension).
        ymin: Uniform prior minimum y edge (second dimension).
        ymax: Uniform prior maximum y edge (second dimension).

    Returns:
        double: Posterior at specified point.
    """

    ln_L = ln_likelihood(x)

    if not np.isfinite(ln_L):
        return -np.inf
    else:
        return ln_prior(x, xmin, xmax, ymin, ymax) + ln_L


def run_example(ndim=2, nchains=100, samples_per_chain=1000,
                nburn=500, verbose=True,
                plot_corner=False, plot_surface=False):
    """Run Himmelblau example.

    Args:
        ndim: Dimension.
        nchains: Number of chains.
        samples_per_chain: Number of samples per chain.
        nburn: Number of burn in samples.
        plot_corner: Plot marginalised distributions if true.
        plot_surface: Plot surface and samples if true.
        verbose: If True then display intermediate results.

    Returns:
        None.
    """

    print("Himmelblau example")
    print("ndim = {}".format(ndim))
    if ndim != 2:
        raise ValueError("Only ndim=2 is supported (ndim={} specified)"
            .format(ndim))

    # Set parameters.
    savefigs = True
    nfold = 2
    nhyper = 2
    step = -2
    domain = []
    hyper_parameters = [[10**(R)] for R in range(-nhyper+step,step)]
    if verbose: print("hyper_parameters = {}".format(hyper_parameters))
    xmin = -5.0
    xmax = 5.0
    ymin = -5.0
    ymax = 5.0
    if verbose: print("xmin, xmax, ymin, ymax = {}, {}, {}, {}"
        .format(xmin, xmax, ymin, ymax))

    # Start timer.
    clock = time.clock()

    # Run multiple realisations.
    n_realisations = 100
    evidence_inv_summary = np.zeros((n_realisations,3))
    for i_realisation in range(n_realisations):

        if n_realisations > 0:
            print("**** i_realisation = {} ****".format(i_realisation))

        # Set up and run sampler.
        print("Run sampling...")
        pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 10.0 - 5.0
        sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior,
                                        args=[xmin, xmax, ymin, ymax])
        rstate = np.random.get_state()
        sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
        samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
        lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

        # Calculate evidence using harmonic....

        # Set up chains.
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)
        chains_train, chains_test = hm.utils.split_data(chains,
                                                        training_proportion=0.5)

        # Perform cross-validation.
        print("Perform cross-validation...")
        validation_variances = \
            hm.utils.cross_validation(chains_train, \
                                      domain, \
                                      hyper_parameters, \
                                      nfold=nfold, \
                                      modelClass=hm.model.KernelDensityEstimate, \
                                      verbose=verbose, \
                                      seed=0)
        if verbose: print("validation_variances = {}".format(validation_variances))
        best_hyper_param_ind = np.argmin(validation_variances)
        best_hyper_param = hyper_parameters[best_hyper_param_ind]
        if verbose: print("best_hyper_param = {}".format(best_hyper_param))

        # Fit model.
        print("Fit model...")
        model = hm.model.KernelDensityEstimate(ndim,
                                               domain,
                                               hyper_parameters=best_hyper_param)
        fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
        if verbose: print("fit_success = {}".format(fit_success))

        # Use chains and model to compute evidence.
        print("Compute evidence...")
        ev = hm.Evidence(chains_test.nchains, model)
        ev.add_chains(chains_test)
        ln_evidence, ln_evidence_std = ev.compute_ln_evidence()

        # Compute analytic evidence.
        if ndim == 2 and i_realisation == 0:
            print("Compute evidence by high-resolution numerical integration...")
            ln_posterior_func = partial(ln_posterior, xmin=xmin, xmax=xmax,
                                        ymin=ymin, ymax=ymax)
            ln_posterior_grid, x_grid, y_grid = \
                utils.eval_func_on_grid(ln_posterior_func,
                                        xmin=xmin, xmax=xmax,
                                        ymin=ymin, ymax=ymax,
                                        nx=1000, ny=1000)
            dx = x_grid[0,1] - x_grid[0,0]
            dy = y_grid[1,0] - y_grid[0,0]
            evidence_numerical_integration = np.sum(np.exp(ln_posterior_grid)) * dx * dy
            if verbose: print("dx = {}".format(dx))
            if verbose: print("dy = {}".format(dy))

        # Display results.
        print("evidence_numerical_integration = {}"
            .format(evidence_numerical_integration))
        print("evidence = {}".format(np.exp(ln_evidence)))
        print("evidence_std = {}".format(np.exp(ln_evidence_std)))
        print("evidence_std / evidence = {}"
              .format(np.exp(ln_evidence_std - ln_evidence)))
        diff = np.log(np.abs(evidence_numerical_integration - np.exp(ln_evidence)))
        print("|evidence_numerical_integration - evidence| / evidence = {}"
              .format(np.exp(diff - ln_evidence)))

        if verbose: print("\nevidence_inv_numerical_integration = {}"
            .format(1.0/evidence_numerical_integration))
        if verbose: print("evidence_inv = {}"
            .format(ev.evidence_inv))
        if verbose: print("evidence_inv_std = {}"
            .format(np.sqrt(ev.evidence_inv_var)))
        if verbose: print("evidence_inv_std / evidence_inv = {}"
            .format(np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
        if verbose: print("kurtosis = {}"
            .format(ev.kurtosis))
        if verbose: print("sqrt(2/(n_eff-1)) = {}"
            .format(np.sqrt(2.0/(ev.n_eff-1))))
        if verbose: print("sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var = {}"
            .format(np.sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var))
        if verbose: print(
            "|evidence_inv_numerical_integration - evidence_inv| / evidence_inv = {}"
            .format(np.abs(1.0 / evidence_numerical_integration - ev.evidence_inv)
                    / ev.evidence_inv))

        if verbose: print("\nlnargmax = {}"
            .format(ev.lnargmax))
        if verbose: print("lnargmin = {}"
            .format(ev.lnargmin))
        if verbose: print("lnprobmax = {}"
            .format(ev.lnprobmax))
        if verbose: print("lnprobmin = {}"
            .format(ev.lnprobmin))
        if verbose: print("lnpredictmax = {}"
            .format(ev.lnpredictmax))
        if verbose: print("lnpredictmin = {}"
            .format(ev.lnpredictmin))
        if verbose: print("mean_shift = {}"
            .format(ev.mean_shift))

        if verbose: print("\nrunning_sum = \n{}"
            .format(ev.running_sum))
        if verbose: print("running_sum_total = \n{}"
            .format(sum(ev.running_sum)))

        if verbose: print("\nnsamples_per_chain = \n{}"
            .format(ev.nsamples_per_chain))
        if verbose: print("nsamples_eff_per_chain = \n{}"
            .format(ev.nsamples_eff_per_chain))

        # Create corner/triangle plot.
        created_plots = False
        if plot_corner and i_realisation == 0:

            utils.plot_corner(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('./plots/himmelblau_corner.png',
                            bbox_inches='tight')

            utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('./plots/himmelblau_getdist.png',
                            bbox_inches='tight')

            plt.show(block=False)
            created_plots = True

        # In 2D case, plot surface/image and samples.
        if plot_surface and ndim == 2 and i_realisation == 0:

            # Plot ln_posterior surface.
            i_chain = 0
            ax = utils.plot_surface(ln_posterior_grid, x_grid, y_grid,
                                    samples[i_chain,:,:].reshape((-1, ndim)),
                                    lnprob[i_chain,:].reshape((-1, 1)))
            # ax.set_zlim(-100.0, 0.0)
            ax.set_zlabel(r'$\log \mathcal{L}$')
            if savefigs:
                plt.savefig('./plots/himmelblau_lnposterior_surface.png',
                            bbox_inches='tight')

            # Plot posterior image.
            ax = utils.plot_image(np.exp(ln_posterior_grid), x_grid, y_grid,
                                  samples.reshape((-1,ndim)),
                                  colorbar_label=r'$\mathcal{L}$')
            # ax.set_clim(vmin=0.0, vmax=0.003)
            if savefigs:
                plt.savefig('./plots/himmelblau_posterior_image.png',
                            bbox_inches='tight')

            # Evaluate model on grid.
            model_grid, x_grid, y_grid = \
                utils.eval_func_on_grid(model.predict,
                                        xmin=xmin, xmax=xmax,
                                        ymin=ymin, ymax=ymax,
                                        nx=1000, ny=1000)

            # Plot model.
            ax = utils.plot_image(model_grid, x_grid, y_grid,
                                  colorbar_label=r'$\log \varphi$')
            # ax.set_clim(vmin=-2.0, vmax=2.0)
            if savefigs:
                plt.savefig('./plots/himmelblau_model_image.png',
                            bbox_inches='tight')

            # Plot exponential of model.
            ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                                  colorbar_label=r'$\varphi$')
            # ax.set_clim(vmin=0.0, vmax=10.0)
            if savefigs:
                plt.savefig('./plots/himmelblau_modelexp_image.png',
                            bbox_inches='tight')

            plt.show(block=False)
            created_plots = True

        evidence_inv_summary[i_realisation,0] = ev.evidence_inv
        evidence_inv_summary[i_realisation,1] = ev.evidence_inv_var
        evidence_inv_summary[i_realisation,2] = ev.evidence_inv_var_var

    clock = time.clock() - clock
    print("execution_time = {}s".format(clock))

    if n_realisations > 1:
        np.savetxt("examples/data/himmelblau_evidence_inv" +
                   "_realisations.dat",
                   evidence_inv_summary)
        evidence_inv_analytic_summary = np.zeros(1)
        evidence_inv_analytic_summary[0] = 1.0 / evidence_numerical_integration
        np.savetxt("examples/data/himmelblau_evidence_inv" +
                   "_analytic.dat",
                   evidence_inv_analytic_summary)

    if created_plots:
        input("\nPress Enter to continue...")

    return samples


if __name__ == '__main__':

    # Define parameters.
    ndim = 2
    nchains = 200
    samples_per_chain = 5000
    nburn = 2000
    np.random.seed(20)

    # Run example.
    samples = run_example(ndim, nchains, samples_per_chain, nburn,
                          plot_corner=True, plot_surface=True, verbose=True)
