import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
import emcee
import scipy.special as sp
import time 
import matplotlib.pyplot as plt

def ln_Prior(x, sigma=5.):

    return -0.5*np.dot(x-1,x-1)/sigma**2 - 0.5*x.size*np.log(2*np.pi*sigma)

def ln_Likelihood(x, a=100.):
    ndim = x.size

    f = 0.0

    for i_dim in range(ndim-1):
        f += a*(x[i_dim+1]-x[i_dim]**2)**2 + (1-x[i_dim])**2

    return -f

def ln_Posterior(x, a=100., sigma=50.):
    ln_L = ln_Likelihood(x, a=a)

    if not np.isfinite(ln_L):
        return -np.inf
    else:
        return ln_Prior(x, sigma=sigma) + ln_L


print("Rosenbrock example:")

ndim = 2

nchains               = 200
samples_per_chain     = 3000
burn_in               = 2000
samples_per_chain_net = (samples_per_chain-burn_in)

plot_sample = False
verbose     = True

nfold = 2

nhyper = 2
step   = -2
domain = []
hyper_parameters = [[10**(R)] for R in range(-nhyper+step,step)]
print("hyper parameters to try : ", hyper_parameters)
n_real = 1


for i_real in range(n_real):

    pos = [np.random.randn(ndim)*0.1 for i in range(nchains)]

    sampler = emcee.EnsembleSampler(nchains, ndim, ln_Posterior)
    sampler.run_mcmc(pos, samples_per_chain)

    samples = np.ascontiguousarray(sampler.chain[:,burn_in:,:])
    Y = np.ascontiguousarray(sampler.lnprobability[:,burn_in:])
    print("samples drawn")

    if plot_sample:
        plt.plot(sampler.chain[0,:,0])
        plt.plot(sampler.chain[0,:,1])
        plt.show()

        import corner
        fig = corner.corner(samples.reshape((-1, ndim)))#, labels=["x1","x2","x3","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4"],
                         #truths=[0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        plt.show()


    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, Y)

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



