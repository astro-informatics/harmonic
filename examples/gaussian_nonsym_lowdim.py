import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
import emcee
import scipy.special as sp
import time 
import matplotlib.pyplot as plt

def ln_analytic_evidence(ndim, cov):
    ln_norm_lik = -0.5*ndim*np.log(2*np.pi)-0.5*np.log(np.linalg.det(cov))
    return -ln_norm_lik

def ln_Posterior(x, inv_cov):
    return -np.dot(x,np.dot(inv_cov,x))/2.0

print("nD Guassian example:")

ndim = 5
print("ndim = ", ndim)


cov = np.zeros((ndim,ndim))
diag_cov = np.ones(ndim) + np.random.randn(ndim)*0.1
np.fill_diagonal(cov, diag_cov)

cov[0,1] = 0.5*np.sqrt(cov[0,0]*cov[1,1])
cov[1,0] = 0.5*np.sqrt(cov[0,0]*cov[1,1])
inv_cov = np.linalg.inv(cov)

print("With covarience matrix = \n", cov)


ln_rho = -ln_analytic_evidence(ndim, cov)
print("ln_rho = ", ln_rho)


nchains               = 200
samples_per_chain     = 1000
burn_in               = 0
samples_per_chain_net = (samples_per_chain-burn_in)

plot_sample = False

n_real = 1

max_r_prob = np.sqrt(ndim-1)
print("max_r_prob = ", max_r_prob)

domains = [max_r_prob*np.array([1E-1,1E1])]



clock = time.clock()
for i_real in range(n_real):
    print("i_real : ", i_real)
    pos = [np.zeros(ndim) + np.random.randn(ndim) for i in range(nchains)]

    sampler = emcee.EnsembleSampler(nchains, ndim, ln_Posterior, args=[inv_cov])
    (pos, prob, state) = sampler.run_mcmc(pos, samples_per_chain)

    samples = np.ascontiguousarray(sampler.chain[:,burn_in:,:])
    Y = np.ascontiguousarray(sampler.lnprobability[:,burn_in:])

    if plot_sample:
        import corner
        fig = corner.corner(samples.reshape((-1, ndim)))#, labels=["x1","x2","x3","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4"],
                         #truths=[0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        plt.show()

    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, Y)

    chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.01)

    sphere = hm.model.HyperSphere(ndim, domains)
    sphere.fit(chains_train.samples,chains_train.ln_posterior)

    cal_ev = hm.Evidence(chains_test.nchains, sphere)
    cal_ev.add_chains(chains_test)

    print("ln_rho_est = ", np.log(cal_ev.evidence_inv), \
        " rel error = ", np.sqrt(cal_ev.evidence_inv_var)/cal_ev.evidence_inv, "(in linear space)")
        
        
    # Temporary code to add more chains    
    sampler.reset()
    (pos, prob, state) = sampler.run_mcmc(pos, 10*samples_per_chain)
    samples = np.ascontiguousarray(sampler.chain[:,:,:])
    Y = np.ascontiguousarray(sampler.lnprobability[:,:])

    
    chains2 = hm.Chains(ndim)
    chains2.add_chains_3d(samples, Y)
    
    chains_train2, chains_test2 = hm.utils.split_data(chains2, training_proportion=0.01)
    
    cal_ev.add_chains(chains_test2)

    
    print("ln_rho_est = ", np.log(cal_ev.evidence_inv), \
        " rel error = ", np.sqrt(cal_ev.evidence_inv_var)/cal_ev.evidence_inv, "(in linear space)")
            

clock = time.clock() - clock


