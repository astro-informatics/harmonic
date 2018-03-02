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

nchains               = 100
samples_per_chain     = 3000
burn_in               = 2000
samples_per_chain_net = (samples_per_chain-burn_in)

plot_sample = False

n_real = 1

ln_rho = np.log(1000.)
print("ln_rho = ", ln_rho)

nguassians = [15,20,25,30,35]
domains    = [np.array([5E-2,5E0])]

np.random.seed(0)
pos = [np.random.randn(ndim)*0.1 for i in range(nchains)]

sampler = emcee.EnsembleSampler(nchains, ndim, ln_Posterior)
sampler.run_mcmc(pos, samples_per_chain)

samples = np.ascontiguousarray(sampler.chain[:,burn_in:,:])
Y = np.ascontiguousarray(sampler.lnprobability[:,burn_in:])

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

hyper_parameters = [nguassians[2],1E-4,0.1,10000,2]
MGMM = hm.model.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=hyper_parameters)
MGMM.verbose=True
MGMM.fit(chains.samples,chains.ln_posterior)

hyper_parameters_MGMM = [[nguassians,1E-30,0.1*nguassians*nguassians,10000,100] for nguassians in range(1,4)]

validation_variances = utils.cross_validation(chains, 
    [np.array([1E-2,10E0])], \
    hyper_parameters_MGMM, modelClass=md.ModifiedGaussianMixtureModel, verbose=True)

chains = hm.Chains(ndim)
chains.add_chains_3d(samples, Y)

sphere = hm.model.HyperSphere(ndim, domains)
sphere.fit(chains.samples,chains.ln_posterior)

cal_ev = hm.Evidence(nchains, sphere)
cal_ev.add_chains(chains)

print("ln_rho_est = ", np.log(cal_ev.evidence_inv), \
    " rel error = ", np.sqrt(cal_ev.evidence_inv_var)/cal_ev.evidence_inv, "(in linear space)")



