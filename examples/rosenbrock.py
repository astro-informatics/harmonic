import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
import emcee
import time 
import matplotlib.pyplot as plt
from functools import partial
sys.path.append("examples")
import utils


def ln_prior(x, mu=1.0, sigma=5.):
    """Compute log_e of Gaussian prior.

    Args: 
        x: Position at which to evaluate prior.
        mu: Mean (centre) of the prior.
        sigma: Standard deviation of prior.   
        
    Returns:
        double: Value of prior at specified point.
    """
    
    return - 0.5 * np.dot(x-mu, x-mu) / sigma**2 \
           - 0.5 * x.size * np.log(2 * np.pi * sigma)


def ln_likelihood(x, a=1.0, b=100.0):
    """Compute log_e of likelihood defined by Rosenbrock function.
    
    Args: 
        x: Position at which to evaluate likelihood.
        a: First parameter of Rosenbrock function.   
        b: First parameter of Rosenbrock function.
        
    Returns:
        double: Value of Rosenbrock at specified point.
    """
    
    ndim = x.size

    f = 0.0

    for i_dim in range(ndim-1):
        f += b*(x[i_dim+1]-x[i_dim]**2)**2 + (a-x[i_dim])**2

    return -f


def ln_posterior(x, a=1.0, b=100.0, mu=1.0, sigma=50.):
    """Compute log_e of posterior.
    
    Args: 
        x: Position at which to evaluate posterior.
        a: First parameter of Rosenbrock function.   
        b: First parameter of Rosenbrock function.
        mu: Mean (centre) of the prior.
        sigma: Standard deviation of prior.
        
    Returns:
        double: Posterior at specified point.
    """
    
    ln_L = ln_likelihood(x, a=a, b=b)

    if not np.isfinite(ln_L):
        return -np.inf
    else:
        return ln_prior(x, mu=mu, sigma=sigma) + ln_L


def run_example(ndim=2, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False):
    """Run Rosenbrock example.

    Args: 
        ndim: Dimension of Gaussian.
        nchains: Number of chains.
        samples_per_chain: Number of samples per chain.
        nburn: Number of burn in samples.
        plot_corner: Plot marginalised distributions if true.
        plot_surface: Plot surface and samples if true.
        verbose: If True then display intermediate results.
        
    Returns:
        None.
    """
    
    print("Rosenbrock example")
    print("ndim = {}".format(ndim))

    # Set parameters.
    savefigs = False
    nfold = 2
    nhyper = 2
    step = -2
    domain = []
    hyper_parameters = [[10**(R)] for R in range(-nhyper+step,step)]
    if verbose: print("hyper_parameters = {}".format(hyper_parameters))
    a = 1.0
    b = 100.0
    mu = 1.0
    sigma = 50.0
    if verbose: print("a, b, mu, sigma = {}, {}, {}, {}"
        .format(a, b, mu, sigma))

    # Start timer.
    clock = time.clock()
    
    # Set up and run sampler.
    print("Run sampling...")
    pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 0.1
    sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=[a, b, mu, sigma])
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
    
    # Use chains and model to compute inverse evidence.
    print("Compute evidence...")
    ev = hm.Evidence(chains_test.nchains, model)    
    ev.add_chains(chains_test)
    ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
    
    # Compute analytic evidence.
    if ndim == 2:
        print("Compute evidence by high-resolution numerical integration...")
        ln_posterior_func = partial(ln_posterior, a=a, b=b, mu=mu, sigma=sigma)
        ln_posterior_grid, x_grid, y_grid = \
            utils.eval_func_on_grid(ln_posterior_func, 
                                    xmin=-10.0, xmax=10.0, 
                                    ymin=-5.0, ymax=15.0, 
                                    nx=500, ny=500)
        dx = x_grid[0,1] - x_grid[0,0]
        dy = y_grid[1,0] - y_grid[0,0]
        evidence_numerical_integration = np.sum(np.exp(ln_posterior_grid)) * dx * dy
        if verbose: print("dx = {}".format(dx))
        if verbose: print("dy = {}".format(dy))
        print("evidence_numerical_integration = {}"
            .format(evidence_numerical_integration))
    
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
    
    if verbose: print("\nevidence_inv_analytic = {}"
        .format(1.0/evidence_numerical_integration))
    if verbose: print("evidence_inv = {}"
        .format(ev.evidence_inv))
    if verbose: print("evidence_inv_std = {}"
        .format(np.sqrt(ev.evidence_inv_var)))
    if verbose: print("evidence_inv_std / evidence_inv = {}"
        .format(np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
    
    if verbose: print(
        "|evidence_numerical_integration_inv - evidence_inv| / evidence_inv = {}"
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
    
    
    
    
    # TODO: tidy up printing in cross validation DONE
    # TODO: plotting
    # TODO: evidence calculation by numerical integration DONE
    # TODO: consider uniform prior?
    # TODO: extend to higher dimensions?
    



    # Create corner/triangle plot.
    if plot_corner:
        
        utils.plot_corner(samples.reshape((-1, ndim)))
        if savefigs:
            plt.savefig('./plots/rosenbrock_corner.png',
                        bbox_inches='tight')
        
        utils.plot_getdist(samples.reshape((-1, ndim)))
        if savefigs:
            plt.savefig('./plots/rosenbrock_getdist.png',
                        bbox_inches='tight')
        
        plt.show()  
            
    # In 2D case, plot surface/image and samples.    
    if plot_surface and ndim == 2:
        
        
        
        # ln_posterior_grid[ln_posterior_grid<-100.0] = -100.0
        
        # ax = utils.plot_surface(ln_posterior_grid, x_grid, y_grid)
        i_chain = 0
        ax = utils.plot_surface(ln_posterior_grid, x_grid, y_grid, samples[i_chain,:,:].reshape((-1, ndim)), lnprob[i_chain,:].reshape((-1, 1)))
        # ax.set_zlim(-100.0, 0.0)
        # ax.set_zlabel('$ln L$')
        
        
        
        
        ax = utils.plot_image(np.exp(ln_posterior_grid), x_grid, y_grid, samples.reshape((-1,ndim)))

        func_eval_grid, x_grid, y_grid = \
            utils.eval_func_on_grid(model.predict, 
                                    xmin=-3.0, xmax=3.0, 
                                    ymin=-2.0, ymax=12.0, 
                                    nx=1000, ny=1000)

        ax = utils.plot_image((func_eval_grid), x_grid, y_grid)

        ax = utils.plot_image(np.exp(func_eval_grid), x_grid, y_grid)


        

        # func_eval_grid[func_eval_grid<-100.0] = -100.0        



        # func_eval_grid, x_grid, y_grid = \
        #     utils.eval_func_on_grid(model.predict, 
        #                             xmin=-3.0, xmax=3.0, 
        #                             ymin=-2.0, ymax=7.0, 
        #                             nx=500, ny=500)
        # 
        # ax = utils.plot_surface(func_eval_grid, x_grid, y_grid)
        # # ax.set_zlim(-20.0, 0.0)
        # ax.set_zlabel('$ln L$')
        # 
        # ax = utils.plot_image(func_eval_grid, x_grid, y_grid)

        # ax.set_vmin(-10.0)
        # ax.set_vmax(1.0)

        # ax.set_clim(vmin=-3.0) 
        # ax.set_clim(vmax=10.0) 

        # 
        plt.show()  


    # plot_sample = False

    # 
    # print("samples drawn")
    # 
    # if plot_sample:
    #     plt.plot(sampler.chain[0,:,0])
    #     plt.plot(sampler.chain[0,:,1])
    #     plt.show()
    # 
    #     import corner
    #     fig = corner.corner(samples.reshape((-1, ndim)))
    #     plt.show()


    clock = time.clock() - clock
    print("execution_time = {}s".format(clock))

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
                plot_corner=False, plot_surface=True, verbose=True)
                

