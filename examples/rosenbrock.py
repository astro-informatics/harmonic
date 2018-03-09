import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
import emcee
import time 
import matplotlib.pyplot as plt


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
#TODO: why minus f?


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
                    
    print("Rosenbrock example")
    print("ndim = {}".format(ndim))

    



    plot_sample = False

    nfold = 2

    nhyper = 2
    step   = -2
    domain = []
    hyper_parameters = [[10**(R)] for R in range(-nhyper+step,step)]
    print("hyper parameters to try : ", hyper_parameters)
    n_real = 1

    a = 1.0
    b = 100.0
    mu = 1.0
    sigma = 50.0


    for i_real in range(n_real):

        # Set up and run sampler.
        pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 0.1
        
        sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=[a, b, mu, sigma])
        rstate = np.random.get_state()
        sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
        samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
        lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])






        print("samples drawn")

        if plot_sample:
            plt.plot(sampler.chain[0,:,0])
            plt.plot(sampler.chain[0,:,1])
            plt.show()

            import corner
            fig = corner.corner(samples.reshape((-1, ndim)))
            plt.show()


        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)

        chains_trian, chains_test = hm.utils.split_data(chains)

        print("start validation")
        validation_variances = hm.utils.cross_validation(chains_trian, 
            domain, \
            hyper_parameters, nfold=nfold, modelClass=hm.model.KernelDensityEstimate, verbose=verbose, seed=0)

        print("validation variances: ", validation_variances)
        best_hyper_param = np.argmin(validation_variances)

        print("Using hyper parameter ", hyper_parameters[best_hyper_param])

        density = hm.model.KernelDensityEstimate(ndim, domain, hyper_parameters=hyper_parameters[best_hyper_param])

        density.fit(chains_trian.samples,chains_trian.ln_posterior)

        cal_ev = hm.Evidence(chains_test.nchains, density)
        cal_ev.add_chains(chains_test)

        ln_rho = np.log(1000.) # Analytic for 2D.
        print("ln_rho = ", ln_rho)
        print("ln_rho_est = ", np.log(cal_ev.evidence_inv), \
            " rel error = ", np.sqrt(cal_ev.evidence_inv_var)/cal_ev.evidence_inv, "(in linear space)")




if __name__ == '__main__':
    
    # Define parameters.
    ndim = 2
    nchains = 200
    samples_per_chain = 3000
    nburn = 2000
    np.random.seed(20)
    
    # Run example.
    run_example(ndim, nchains, samples_per_chain, nburn, 
                plot_corner=False, plot_surface=False, verbose=False)