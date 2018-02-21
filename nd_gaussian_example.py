import numpy as np
import chains as ch
import calculate_evidence as cbe
import model as md
import emcee
import scipy.special as sp
import time 
import matplotlib.pyplot as plt

def ln_analytic_evidence(ndim, cov):
    # ln_vol = (ndim/2)*np.log(np.pi) + ndim*np.log(w) - sp.gammaln(ndim/2+1)
	ln_norm_lik = -0.5*ndim*np.log(2*np.pi)-0.5*np.log(np.linalg.det(cov))
	return -ln_norm_lik

def ln_Posterior(x, inv_cov):
	return -np.dot(x,np.dot(inv_cov,x))/2.0

def X_to_Y(X, inv_cov):
	Y = np.empty((X.shape[0],X.shape[1]))
	for i_x in range(X.shape[0]):
		for i_y in range(X.shape[1]):
			Y[i_x,i_y] = ln_Posterior(X[i_x,i_y,:], inv_cov)
	return Y

ndim = 5
print("ndim = ", ndim)


cov = np.zeros((ndim,ndim))
diag_cov = np.ones(ndim)
np.fill_diagonal(cov, diag_cov)

# cov[0,1] = 0.5
# cov[1,0] = 0.5
inv_cov = np.linalg.inv(cov)



ln_rho = -ln_analytic_evidence(ndim, cov)
print("ln_rho = ", ln_rho)


nchains               = 200
samples_per_chain     = 10000
burn_in               = 0
samples_per_chain_net = (samples_per_chain-burn_in)

n_real = 1

max_r_prob = np.sqrt(ndim-1)
print("max_r_prob = ", max_r_prob)

domains = [max_r_prob*np.array([1E-1,1E1])]



clock = time.clock()
for i_real in range(n_real):
    print("i_real : ", i_real)
    pos = [np.zeros(ndim) + np.random.randn(ndim) for i in range(nchains)]

    sampler = emcee.EnsembleSampler(nchains, ndim, ln_Posterior, args=[inv_cov])
    sampler.run_mcmc(pos, samples_per_chain)

    # if i_real is 0:
    #     plt.plot(sampler.chain[0,:,0])
    #     plt.plot(sampler.chain[0,:,1])
    #     plt.plot(sampler.chain[0,:,2])
    #     plt.show()

    samples = sampler.chain[:,burn_in:,:]
    Y       = X_to_Y(samples,inv_cov)

    # import corner
    # fig = corner.corner(samples.reshape((-1, ndim)))#, labels=["x1","x2","x3","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4"],
    #                  #truths=[0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    # plt.show()

    chains = ch.Chains(ndim)
    chains.add_chains_3d(samples, Y)

    sphere = md.HyperSphere(ndim, domains)
    sphere.fit(chains.samples,chains.ln_posterior)

    cal_ev = cbe.evidence(nchains)
    cal_ev.calculate_evidence(chains,sphere)

    print(np.exp(ln_rho), cal_ev.p, np.sqrt(cal_ev.s2)/cal_ev.p, cal_ev.s2, cal_ev.v2)

clock = time.clock() - clock


