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

# Setup Logging config
hm.logs.setup_logging()

def ln_prior_uniform(x, xmin=-10.0, xmax=10.0, ymin=-5.0, ymax=15.0):
    """
    .. note:: Compute log_e of uniform prior.

    Args: 
        - x: 
            Position at which to evaluate prior.
        - xmin: 
            Uniform prior minimum x edge (first dimension).
        - xmax: 
            Uniform prior maximum x edge (first dimension).
        - ymin: 
            Uniform prior minimum y edge (second dimension).
        - ymax: 
            Uniform prior maximum y edge (second dimension).             
        
    Returns:
        - double: 
            Value of prior at specified point.
    """
        
    if x[0] >= xmin and x[0] <= xmax and x[1] >= ymin and x[1] <= ymax:        
        return 1.0 / ( (xmax - xmin) * (ymax - ymin) )
    else:
        return 0.0
        
        
def ln_prior_gaussian(x, mu=1.0, sigma=5.):
    """
    .. note:: Compute log_e of Gaussian prior.

    Args: 
        - x: 
            Position at which to evaluate prior.
        - mu: 
            Mean (centre) of the prior.
        - sigma: 
            Standard deviation of prior.   
        
    Returns:
        - double: 
            Value of prior at specified point.
    """
        
    return - 0.5 * np.dot(x-mu, x-mu) / sigma**2 \
           - 0.5 * x.size * np.log(2 * np.pi * sigma)


def ln_likelihood(x, a=1.0, b=100.0):
    """
    .. note:: Compute log_e of likelihood defined by Rosenbrock function.
    
    Args: 
        - x: 
            Position at which to evaluate likelihood.
        - a: 
            First parameter of Rosenbrock function.   
        - b: 
            First parameter of Rosenbrock function.
        
    Returns:
        - double: 
            Value of Rosenbrock at specified point.
    """
    
    ndim = x.size

    f = 0.0

    for i_dim in range(ndim-1):
        f += b*(x[i_dim+1]-x[i_dim]**2)**2 + (a-x[i_dim])**2

    return -f


def ln_posterior(x, ln_prior, a=1.0, b=100.0):
    """
    .. note:: Compute log_e of posterior.
    
    Args: 
        - x: 
            Position at which to evaluate posterior.
        - a: 
            First parameter of Rosenbrock function.   
        - b: 
            First parameter of Rosenbrock function.
        - ln_prior: 
            Prior function.
        
    Returns:
        - double: 
            Posterior at specified point.
    """
    
    ln_L = ln_likelihood(x, a=a, b=b)

    if not np.isfinite(ln_L):
        return -np.inf
    else:
        return ln_prior(x) + ln_L


def run_example(ndim=2, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False):
    """
    .. note:: Run Rosenbrock example.

    Args: 
        - ndim: 
            Dimension.
        - nchains: 
            Number of chains.
        - samples_per_chain: 
            Number of samples per chain.
        - nburn: 
            Number of burn in samples.
        - plot_corner: 
            Plot marginalised distributions if true.
        - plot_surface: 
            Plot surface and samples if true.
        - verbose: 
            If True then display intermediate results.
        
    Returns:
        - None.
    """
    
    hm.logs.high_log('Rosenbrock example')
    hm.logs.high_log('Dimensionality = {}'.format(ndim)) 
    hm.logs.low_log('---------------------------------')   
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
    hm.logs.low_log('Hyper-parameters = {}'.format(hyper_parameters))
    a = 1.0
    b = 100.0
    
    # Set prior parameters.
    use_uniform_prior = True
    if use_uniform_prior:        
        xmin = -10.0
        xmax = 10.0
        ymin = -5.0
        ymax = 15.0
        hm.logs.low_log('xmin, xmax, ymin, ymax = {}, {}, {}, {}'
            .format(xmin, xmax, ymin, ymax))   
        ln_prior = partial(ln_prior_uniform, xmin=xmin, xmax=xmax, 
                           ymin=ymin, ymax=ymax)     
    else: # Use Gaussian prior
        mu = 1.0
        sigma = 50.0
        hm.logs.low_log('a, b, mu, sigma = {}, {}, {}, {}'
            .format(a, b, mu, sigma))   
        ln_prior = partial(ln_prior_gaussian, mu=mu, sigma=sigma)
    hm.logs.low_log('---------------------------------')
    # Start timer.
    clock = time.clock()

    # Set up and run multiple simulations
    n_realisations = 100
    evidence_inv_summary = np.zeros((n_realisations,3))
    for i_realisation in range(n_realisations):

        if n_realisations > 1:
            hm.logs.high_log('Realisation number = {}/{}'.format(i_realisation, n_realisations))
        
        # Set up and run sampler.
        hm.logs.high_log('Run sampling...')
        pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 0.1    
        sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, 
                                        args=[ln_prior, a, b])
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
        hm.logs.high_log('Perform cross-validation...')
        hm.logs.low_log('---------------------------------')
        validation_variances = \
            hm.utils.cross_validation(chains_train, \
                                      domain, \
                                      hyper_parameters, \
                                      nfold=nfold, \
                                      modelClass=hm.model.KernelDensityEstimate, \
                                      verbose=verbose, \
                                      seed=0)
        hm.logs.low_log('validation_variances = {}'.format(validation_variances))
        best_hyper_param_ind = np.argmin(validation_variances)
        best_hyper_param = hyper_parameters[best_hyper_param_ind]
        hm.logs.low_log('Best hyper-parameter = {}'.format(best_hyper_param))
        hm.logs.low_log('---------------------------------')

        # Fit model.
        hm.logs.high_log('Fit model...')
        hm.logs.low_log('---------------------------------')
        model = hm.model.KernelDensityEstimate(ndim, 
                                               domain, 
                                               hyper_parameters=best_hyper_param)
        fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
        hm.logs.low_log('Fit success = {}'.format(fit_success))    
        hm.logs.low_log('---------------------------------')

        # Use chains and model to compute evidence.
        hm.logs.high_log('Compute evidence...')
        ev = hm.Evidence(chains_test.nchains, model)    
        ev.add_chains(chains_test)
        ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
        
        # Compute analytic evidence.
        if ndim == 2:
            hm.logs.high_log('Compute evidence by high-resolution numerical integration...')
            hm.logs.low_log('---------------------------------')
            ln_posterior_func = partial(ln_posterior, ln_prior=ln_prior, a=a, b=b)
            ln_posterior_grid, x_grid, y_grid = \
                utils.eval_func_on_grid(ln_posterior_func, 
                                        xmin=-10.0, xmax=10.0, 
                                        ymin=-5.0, ymax=15.0, 
                                        nx=1000, ny=1000)
            dx = x_grid[0,1] - x_grid[0,0]
            dy = y_grid[1,0] - y_grid[0,0]
            evidence_numerical_integration = np.sum(np.exp(ln_posterior_grid)) * dx * dy
            hm.logs.low_log('dx = {}'.format(dx))
            hm.logs.low_log('dy = {}'.format(dy))    
        
        # ===============================================================================
        # Display evidence computation results.
        # ===============================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('Evidence: numerical = {}, estimate = {}'
            .format(evidence_numerical_integration, np.exp(ln_evidence)))
        hm.logs.low_log('Evidence: std = {}, std / estimate = {}'
            .format(np.exp(ln_evidence_std), np.exp(ln_evidence_std - ln_evidence)))
        diff = np.log(np.abs(evidence_numerical_integration - np.exp(ln_evidence)))
        hm.logs.high_log('Evidence: |numerical - estimate| / estimate = {}'
            .format(np.exp(diff - ln_evidence)))
        # ===============================================================================
        # Display inverse evidence computation results.
        # ===============================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('Inv Evidence: numerical = {}, estimate = {}'
            .format(1.0/evidence_numerical_integration, ev.evidence_inv))
        hm.logs.low_log('Inv Evidence: std = {}, std / estimate = {}'
            .format(np.sqrt(ev.evidence_inv_var), np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
        hm.logs.low_log('Inv Evidence: kurtosis = {}, sqrt( 2 / ( n_eff - 1 ) ) = {}'
            .format(ev.kurtosis, np.sqrt(2.0/(ev.n_eff-1))))    
        hm.logs.low_log('Inv Evidence: sqrt( var(var) )/ var = {}'
            .format(np.sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var))    
        hm.logs.high_log('Inv Evidence: |numerical - estimate| / estimate = {}'
            .format(np.abs(1.0 / evidence_numerical_integration - ev.evidence_inv) / ev.evidence_inv))
        # ===============================================================================
        # Display more technical details for ln evidence.
        # ===============================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('lnargmax = {}, lnargmin = {}'
            .format(ev.lnargmax, ev.lnargmin))
        hm.logs.low_log('lnprobmax = {}, lnprobmin = {}'
            .format(ev.lnprobmax, ev.lnprobmin))
        hm.logs.low_log('lnpredictmax = {}, lnpredictmin = {}'
            .format(ev.lnpredictmax, ev.lnpredictmin))
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('mean shift = {}, running sum total = {}'
            .format(ev.mean_shift, sum(ev.running_sum)))
        hm.logs.low_log('running sum = \n{}'
            .format(ev.running_sum))
        hm.logs.low_log('nsamples per chain = \n{}'
            .format(ev.nsamples_per_chain))
        hm.logs.low_log('nsamples eff per chain = \n{}'
            .format(ev.nsamples_eff_per_chain))
        hm.logs.low_log('===============================')

        # Create corner/triangle plot.
        created_plots = False
        if plot_corner and i_realisation == 0:
            
            utils.plot_corner(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('./plots/rosenbrock_corner.png',
                            bbox_inches='tight')
            
            utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('./plots/rosenbrock_getdist.png',
                            bbox_inches='tight')
            
            plt.show(block=False)  
            created_plots = True
                
        # In 2D case, plot surface/image and samples.    
        if plot_surface and ndim == 2 and i_realisation == 0:
            
            # Plot ln_posterior surface.
            # ln_posterior_grid[ln_posterior_grid<-100.0] = -100.0 
            i_chain = 0
            ax = utils.plot_surface(ln_posterior_grid, x_grid, y_grid, 
                                    samples[i_chain,:,:].reshape((-1, ndim)), 
                                    lnprob[i_chain,:].reshape((-1, 1)))
            # ax.set_zlim(-100.0, 0.0)                
            ax.set_zlabel(r'$\log \mathcal{L}$')        
            if savefigs:
                plt.savefig('./plots/rosenbrock_lnposterior_surface.png',
                            bbox_inches='tight')
            
            # Plot posterior image.
            ax = utils.plot_image(np.exp(ln_posterior_grid), x_grid, y_grid, 
                                  samples.reshape((-1,ndim)),
                                  colorbar_label=r'$\mathcal{L}$')
            # ax.set_clim(vmin=0.0, vmax=0.003)
            if savefigs:
                plt.savefig('./plots/rosenbrock_posterior_image.png',
                            bbox_inches='tight')

            # Evaluate model on grid.
            model_grid, x_grid, y_grid = \
                utils.eval_func_on_grid(model.predict, 
                                        xmin=-10.0, xmax=10.0, 
                                        ymin=-5.0, ymax=15.0, 
                                        nx=1000, ny=1000)
            # model_grid[model_grid<-100.0] = -100.0 
            
            # Plot model.
            ax = utils.plot_image(model_grid, x_grid, y_grid, 
                                  colorbar_label=r'$\log \varphi$') 
            # ax.set_clim(vmin=-2.0, vmax=2.0)
            if savefigs:
                plt.savefig('./plots/rosenbrock_model_image.png',
                            bbox_inches='tight')
            
            # Plot exponential of model.
            ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                                  colorbar_label=r'$\varphi$')
            # ax.set_clim(vmin=0.0, vmax=10.0)        
            if savefigs:
                plt.savefig('./plots/rosenbrock_modelexp_image.png',
                            bbox_inches='tight')

            plt.show(block=False)  
            created_plots = True

        # Save out realisations for voilin plot.
        evidence_inv_summary[i_realisation,0] = ev.evidence_inv
        evidence_inv_summary[i_realisation,1] = ev.evidence_inv_var
        evidence_inv_summary[i_realisation,2] = ev.evidence_inv_var_var

    clock = time.clock() - clock
    hm.logs.high_log('Execution time = {}s'.format(clock))

    if n_realisations > 1:
        np.savetxt("./data/rosenbrock_evidence_inv" +
                   "_realisations.dat",
                   evidence_inv_summary)
        evidence_inv_analytic_summary = np.zeros(1)
        evidence_inv_analytic_summary[0] = 1.0 / evidence_numerical_integration
        np.savetxt("./data/rosenbrock_evidence_inv" +
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
                          plot_corner=True, plot_surface=True, verbose=False)
