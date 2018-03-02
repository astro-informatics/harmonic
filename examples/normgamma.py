import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
import emcee
import scipy.special as sp
import time 
import matplotlib.pyplot as plt

def ln_Likelyhood(x_info, mu, tau):
    return -0.5*x_info[2]*tau*(x_info[1]+(x_info[0]-mu)**2) - 0.5*(x_info[2])*np.log(2*np.pi) + 0.5*(x_info[2])*np.log(tau)

def ln_Prior(mu, tau, prior_prams):

    if tau < 0:
        return -np.inf

    mu0, lamb, alpha, beta = prior_prams

    ln_Prior_in =  alpha*np.log(beta) + 0.5*np.log(lamb) - sp.gammaln(alpha) - 0.5*np.log(2*np.pi)
    ln_Prior_in += (alpha-0.5)*np.log(tau)
    ln_Prior_in += -beta*tau
    ln_Prior_in += -0.5*lamb*tau*(mu-mu0)**2
    return ln_Prior_in

def ln_Posterior(theta, x_info, prior_prams):
    mu, tau = theta

    ln_Pr = ln_Prior(mu, tau, prior_prams)

    if not np.isfinite(ln_Pr):
        return -np.inf

    ln_L   = ln_Likelyhood(x_info, mu, tau)
 
    return  ln_L +ln_Pr

def ln_analytic_evidence(x, prior_prams):
    mu0, lamb, alpha, beta = prior_prams

    lamb_n  = lamb  + x.size
    alpha_n = alpha + x.size/2
    beta_n  = beta + 0.5*x.size*np.std(x) + lamb*x.size*(np.mean(x)-mu0)**2/(2*(lamb+x.size))


    ln_z  = sp.gammaln(alpha_n) - sp.gammaln(alpha)
    ln_z += alpha*np.log(beta)  - alpha_n*np.log(beta_n)
    ln_z += 0.5*np.log(lamb)    - 0.5*np.log(lamb_n)
    ln_z -= 0.5*x.size*np.log(2*np.pi)

    return ln_z

print("Norm Gamma example:")

x = np.loadtxt("examples/data/norm_dist_numbers_0_1.txt")

x_info = [np.mean(x), np.std(x), x.size]

ndim = 2

nchains               = 200
samples_per_chain     = 1500
burn_in               = 500
samples_per_chain_net = (samples_per_chain-burn_in)

plot_sample = False

n_real = 1

max_r_prob = np.sqrt(ndim-1)

domains = [max_r_prob*np.array([1E-1,1E1])]

lamb_array = [1E-3, 1E-2, 1E-1, 1E0]

for lamb_el in lamb_array:
    prior_prams = (0.0, lamb_el, 1E-3, 1E-3)


    for i_real in range(n_real):
        pos = [np.array([x_info[0],1.0/x_info[1]**2]) + x_info[1]*np.random.randn(ndim)/np.sqrt(x_info[2]) for i in range(nchains)]

        sampler = emcee.EnsembleSampler(nchains, ndim, ln_Posterior, args=(x_info, prior_prams))
        sampler.run_mcmc(pos, samples_per_chain)

        samples = np.ascontiguousarray(sampler.chain[:,burn_in:,:])
        Y = np.ascontiguousarray(sampler.lnprobability[:,burn_in:])

        if plot_sample:
            import corner
            fig = corner.corner(samples.reshape((-1, ndim)))#, labels=["x1","x2","x3","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4"],
                             #truths=[0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            plt.show()

        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, Y)

        chains_trian, chains_test = hm.utils.split_data(chains, training_proportion=0.25)

        hyper_parameters_MGMM = [[1,1E-8,0.1,6,10],\
                [2,1E-8,0.5,6,10]]#, [3,1E-8,2.0,10,10]]

        validation_variances_MGMM = hm.utils.cross_validation(chains_trian, 
            [np.array([1E-2,5E0])], \
            hyper_parameters_MGMM, nfold=3, modelClass=hm.model.ModifiedGaussianMixtureModel, verbose=True, seed=0)

        print("validation variances MGMM: ", validation_variances_MGMM)
        best_hyper_param_MGMM = np.argmin(validation_variances_MGMM)

        hyper_parameters_sphere = [None]
        validation_variances_Sphere = hm.utils.cross_validation(chains_trian, 
            [np.array([1E-2,5E0])], \
            hyper_parameters_sphere, nfold=3, modelClass=hm.model.HyperSphere, verbose=True, seed=0)

        print("validation variances sphere: ", validation_variances_Sphere)
        best_hyper_param_sphere = np.argmin(validation_variances_Sphere)

        if validation_variances_MGMM[best_hyper_param_MGMM] < validation_variances_Sphere[best_hyper_param_sphere]:
            model = hm.model.ModifiedGaussianMixtureModel(ndim, [np.array([1E-1,5E0])], \
                hyper_parameters=hyper_parameters_MGMM[best_hyper_param_MGMM])
            model.verbose=True
        else:
            model = hm.model.HyperSphere(ndim, [np.array([1E-1,5E0])], \
                            hyper_parameters=None)

        model.fit(chains_trian.samples,chains_trian.ln_posterior)

        cal_ev = hm.Evidence(chains_test.nchains, model)
        cal_ev.add_chains(chains_test)

        ln_rho = -ln_analytic_evidence(x, prior_prams)
        print("ln_rho = ", ln_rho)
        print("ln_rho_est = ", np.log(cal_ev.evidence_inv), \
            " rel error = ", np.sqrt(cal_ev.evidence_inv_var)/cal_ev.evidence_inv, "(in linear space)")



