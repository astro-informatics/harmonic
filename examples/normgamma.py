import numpy as np
import sys
import emcee
import scipy.special as sp
import time 
import matplotlib.pyplot as plt
sys.path.append(".")
import harmonic as hm


def ln_likelihood_original(x_info, mu, tau):
    return -0.5*x_info[2]*tau*(x_info[1]+(x_info[0]-mu)**2) - 0.5*(x_info[2])*np.log(2*np.pi) + 0.5*(x_info[2])*np.log(tau)



def ln_likelihood(x_info, mu, tau):
    
    x_mean = x_info[0]
    x_std = x_info[1]
    x_n = x_info[2]
    
    return -0.5 * x_n * tau * (x_std + (x_mean-mu)**2) \
        - 0.5 * x_n * np.log(2 * np.pi) + 0.5 * x_n * np.log(tau)


def ln_prior(mu, tau, prior_prams):

    if tau < 0:
        return -np.inf

    mu_0, tau_0, alpha_0, beta_0 = prior_prams

    ln_pr = alpha_0 * np.log(beta_0) + 0.5 * np.log(tau_0) 
    ln_pr += - sp.gammaln(alpha_0) - 0.5 * np.log(2*np.pi)
    ln_pr += (alpha_0 - 0.5) * np.log(tau)
    ln_pr += -beta_0 * tau
    ln_pr += -0.5 * tau_0 * tau * (mu - mu_0)**2
    
    return ln_pr


def ln_posterior(theta, x_info, prior_prams):
    
    mu, tau = theta

    ln_pr = ln_prior(mu, tau, prior_prams)

    if not np.isfinite(ln_pr):
        return -np.inf

    ln_L = ln_likelihood(x_info, mu, tau)
 
    return  ln_L + ln_pr


def ln_analytic_evidence(x_info, prior_prams):
    
    mu_0, tau_0, alpha_0, beta_0 = prior_prams

    # x_mean = np.mean(x)
    # x_std = np.std(x)
    # x_n = x.size
    x_mean = x_info[0]
    x_std = x_info[1]
    x_n = x_info[2]

    tau_n  = tau_0  + x_n
    alpha_n = alpha_0 + x_n/2
    beta_n  = beta_0 + 0.5 * x_n * x_std \
        + tau_0 * x_n * (x_mean - mu_0)**2 / (2 * (tau_0 + x_n))

    ln_z  = sp.gammaln(alpha_n) - sp.gammaln(alpha_0)
    ln_z += alpha_0 * np.log(beta_0) - alpha_n * np.log(beta_n)
    ln_z += 0.5 * np.log(tau_0) - 0.5 * np.log(tau_n)
    ln_z -= 0.5 * x_n * np.log(2*np.pi)

    return ln_z





np.random.seed(1)

print("Norm Gamma example:")

n_meas = 100
mu_in  = 0.0
tau_in = 1.0
# x = np.loadtxt("examples/data/norm_dist_numbers_0_1.txt")
x = np.random.normal(mu_in, np.sqrt(1/tau_in), (n_meas))

x_info = [np.mean(x), np.std(x), x.size]

ndim = 2

nchains               = 200
samples_per_chain     = 1500
burn_in               = 500
samples_per_chain_net = (samples_per_chain-burn_in)

nfold = 3
hyper_parameters_MGMM = [[1,1E-8,0.1,6,10],\
        [2,1E-8,0.5,6,10]]#, [3,1E-8,2.0,10,10]]
hyper_parameters_sphere = [None]

training_proportion = 0.25
domains_sphere = [np.array([1E-1,5E0])]
domains_MGMM = [np.array([1E-1,5E0])]

plot_sample = False
verbose     = False

n_real = 1

max_r_prob = np.sqrt(ndim-1)

domains = [max_r_prob*np.array([1E-1,1E1])]

lamb_array = [1E-3, 1E-2, 1E-1, 1E0]

for lamb_el in lamb_array:
    prior_prams = (0.0, lamb_el, 1E-3, 1E-3)


    for i_real in range(n_real):
        pos = [np.array([x_info[0],1.0/x_info[1]**2]) + x_info[1]*np.random.randn(ndim)/np.sqrt(x_info[2]) for i in range(nchains)]

        sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=(x_info, prior_prams))
        rstate = np.random.get_state()
        sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)

        samples = np.ascontiguousarray(sampler.chain[:,burn_in:,:])
        Y = np.ascontiguousarray(sampler.lnprobability[:,burn_in:])

        if plot_sample:
            import corner
            fig = corner.corner(samples.reshape((-1, ndim)))#, labels=["x1","x2","x3","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4","x4"],
                             #truths=[0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            plt.show()

        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, Y)

        chains_trian, chains_test = hm.utils.split_data(chains, training_proportion=training_proportion)


        validation_variances_MGMM = hm.utils.cross_validation(chains_trian, 
            domains_MGMM, \
            hyper_parameters_MGMM, nfold=nfold, modelClass=hm.model.ModifiedGaussianMixtureModel, verbose=verbose, seed=0)

        print("validation variances MGMM: ", validation_variances_MGMM)
        best_hyper_param_MGMM = np.argmin(validation_variances_MGMM)

        validation_variances_Sphere = hm.utils.cross_validation(chains_trian, 
            domains_sphere, \
            hyper_parameters_sphere, nfold=nfold, modelClass=hm.model.HyperSphere, verbose=verbose, seed=0)

        print("validation variances sphere: ", validation_variances_Sphere)
        best_hyper_param_sphere = np.argmin(validation_variances_Sphere)

        if validation_variances_MGMM[best_hyper_param_MGMM] < validation_variances_Sphere[best_hyper_param_sphere]:
            print("Using MGMM with hyper_parameters :", hyper_parameters_MGMM[best_hyper_param_MGMM])
            model = hm.model.ModifiedGaussianMixtureModel(ndim, domains_MGMM, \
                hyper_parameters=hyper_parameters_MGMM[best_hyper_param_MGMM])
            model.verbose=verbose
        else:
            print("Using HyperSphere")
            model = hm.model.HyperSphere(ndim, domains_sphere, \
                            hyper_parameters=None)

        model.fit(chains_trian.samples,chains_trian.ln_posterior)

        cal_ev = hm.Evidence(chains_test.nchains, model)
        cal_ev.add_chains(chains_test)

        ln_rho = -ln_analytic_evidence(x_info, prior_prams)
        print("ln_rho = ", ln_rho)
        print("ln_rho_est = ", np.log(cal_ev.evidence_inv), \
            " rel error = ", np.sqrt(cal_ev.evidence_inv_var)/cal_ev.evidence_inv, "(in linear space)")



