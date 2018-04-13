import numpy as np
import sys
import emcee
import scipy.special as sp
import time 
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(".")
import harmonic as hm
sys.path.append("examples")
import utils


# To be removed...
def ln_likelihood_original(x_info, mu, tau):
    return -0.5*x_info[2]*tau*(x_info[1]+(x_info[0]-mu)**2) - 0.5*(x_info[2])*np.log(2*np.pi) + 0.5*(x_info[2])*np.log(tau)



def ln_likelihood(x_mean, x_std, x_n, mu, tau):
    
    return -0.5 * x_n * tau * (x_std + (x_mean-mu)**2) \
        - 0.5 * x_n * np.log(2 * np.pi) + 0.5 * x_n * np.log(tau)


def ln_prior(mu, tau, prior_params):

    if tau < 0:
        return -np.inf

    mu_0, tau_0, alpha_0, beta_0 = prior_params

    ln_pr = alpha_0 * np.log(beta_0) + 0.5 * np.log(tau_0) 
    ln_pr += - sp.gammaln(alpha_0) - 0.5 * np.log(2*np.pi)
    ln_pr += (alpha_0 - 0.5) * np.log(tau)
    ln_pr += -beta_0 * tau
    ln_pr += -0.5 * tau_0 * tau * (mu - mu_0)**2
    
    return ln_pr


def ln_posterior(theta, x_mean, x_std, x_n, prior_params):
    
    mu, tau = theta

    ln_pr = ln_prior(mu, tau, prior_params)

    if not np.isfinite(ln_pr):
        return -np.inf

    ln_L = ln_likelihood(x_mean, x_std, x_n, mu, tau)
 
    return  ln_L + ln_pr


def ln_analytic_evidence(x_mean, x_std, x_n, prior_params):
    
    mu_0, tau_0, alpha_0, beta_0 = prior_params

    tau_n  = tau_0  + x_n
    alpha_n = alpha_0 + x_n/2
    beta_n  = beta_0 + 0.5 * x_n * x_std \
        + tau_0 * x_n * (x_mean - mu_0)**2 / (2 * (tau_0 + x_n))

    ln_z  = sp.gammaln(alpha_n) - sp.gammaln(alpha_0)
    ln_z += alpha_0 * np.log(beta_0) - alpha_n * np.log(beta_n)
    ln_z += 0.5 * np.log(tau_0) - 0.5 * np.log(tau_n)
    ln_z -= 0.5 * x_n * np.log(2*np.pi)

    return ln_z




def run_example(ndim=2, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_comparison=False):
                
    print("Normal-Gamma example")

    # Set parameters.
    n_meas = 100
    mu_in  = 0.0
    tau_in = 1.0
    tau_array = [1E-4, 1E-3, 1E-2, 1E-1, 1E0]
    # tau_array = [1E-4, 1E-2]

    savefigs = True
    
    nfold = 3
    training_proportion = 0.25
    hyper_parameters_MGMM = [[1, 1E-8, 0.1, 6, 10],\
            [2, 1E-8, 0.5, 6, 10]]#, [3,1E-8,2.0,10,10]]
    hyper_parameters_sphere = [None]
    domains_sphere = [np.array([1E-1,5E0])]
    domains_MGMM = [np.array([1E-1,5E0])]

    n_realisations = 1

    # Generate simulations.
    print("Simulate data...")
    x = np.random.normal(mu_in, np.sqrt(1/tau_in), (n_meas))
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_n = x.size
    if verbose: print("x_mean = {}".format(x_mean))
    if verbose: print("x_std = {}".format(x_std))
    if verbose: print("x_n = {}".format(x_n))

    summary = np.empty((len(tau_array), 4), dtype=float)
    created_plots = False
    for i_tau, tau_prior in enumerate(tau_array):
        
        print("Considering tau = {}...".format(tau_prior))
        
        prior_params = (0.0, tau_prior, 1E-3, 1E-3)

        for i_real in range(n_realisations):
            
            # Set up and run sampler.
            pos = [np.array([x_mean, 1.0/x_std**2]) \
                   + x_std * np.random.randn(ndim) / np.sqrt(x_n) \
                   for i in range(nchains)]
            sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, \
                args=(x_mean, x_std, x_n, prior_params))
            rstate = np.random.get_state()
            sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
            samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
            lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

            # Calculate evidence using harmonic....

            # Set up chains.
            chains = hm.Chains(ndim)
            chains.add_chains_3d(samples, lnprob)
            chains_train, chains_test = hm.utils.split_data(chains, \
                training_proportion=training_proportion)

            # Perform cross-validation.
            print("Perform cross-validation...")
            
            validation_variances_MGMM = \
                hm.utils.cross_validation(chains_train, 
                    domains_MGMM, \
                    hyper_parameters_MGMM, \
                    nfold=nfold, 
                    modelClass=hm.model.ModifiedGaussianMixtureModel, \
                    verbose=verbose, seed=0)                
            if verbose: print("validation_variances_MGMM = {}"
                .format(validation_variances_MGMM))
            best_hyper_param_MGMM_ind = np.argmin(validation_variances_MGMM)
            best_hyper_param_MGMM = \
                hyper_parameters_MGMM[best_hyper_param_MGMM_ind]

            validation_variances_sphere = \
                hm.utils.cross_validation(chains_train, 
                    domains_sphere, \
                    hyper_parameters_sphere, nfold=nfold, 
                    modelClass=hm.model.HyperSphere, 
                    verbose=verbose, seed=0)
            if verbose: print("validation_variances_sphere = {}"
                .format(validation_variances_sphere))
            best_hyper_param_sphere_ind = np.argmin(validation_variances_sphere)
            best_hyper_param_sphere = \
                hyper_parameters_sphere[best_hyper_param_sphere_ind]

            # Fit model.
            print("Fit model...")
            best_var_MGMM = \
                validation_variances_MGMM[best_hyper_param_MGMM_ind]
            best_var_sphere = \
                validation_variances_sphere[best_hyper_param_sphere_ind]
            if best_var_MGMM < best_var_sphere:            
                print("Using MGMM with hyper_parameters = {}"
                    .format(best_hyper_param_MGMM))                
                model = hm.model.ModifiedGaussianMixtureModel(ndim, \
                    domains_MGMM, hyper_parameters=best_hyper_param_MGMM)
                model.verbose=False
            else:
                print("Using HyperSphere")
                model = hm.model.HyperSphere(ndim, domains_sphere, \
                    hyper_parameters=best_hyper_param_sphere)
            model.fit(chains_train.samples, chains_train.ln_posterior)

            # Use chains and model to compute evidence.
            print("Compute evidence...")
            ev = hm.Evidence(chains_test.nchains, model)
            ev.add_chains(chains_test)
            ln_evidence, ln_evidence_std = ev.compute_ln_evidence()

            # Compute analytic evidence.
            ln_evidence_analytic = ln_analytic_evidence(x_mean, \
                x_std, x_n, prior_params)
            evidence_analytic = np.exp(ln_evidence_analytic)
            
            # Collate values.
            summary[i_tau, 0] = tau_prior
            summary[i_tau, 1] = ln_evidence_analytic
            summary[i_tau, 2] = ln_evidence
            summary[i_tau, 3] = ln_evidence_std
            
            # Display results.            
            print("ln_evidence_analytic = {}"
                .format(ln_evidence_analytic))
            print("ln_evidence = {}".format(ln_evidence))            
            diff = np.abs(ln_evidence_analytic - ln_evidence)
            print("|ln_evidence_analytic - ln_evidence| / ln_evidence = {}\n"
                  .format(diff/ln_evidence))

            print("evidence_analytic = {}"
                .format(evidence_analytic))
            print("evidence = {}".format(np.exp(ln_evidence)))
            print("evidence_std = {}".format(np.exp(ln_evidence_std)))
            print("evidence_std / evidence = {}"
                  .format(np.exp(ln_evidence_std - ln_evidence)))
            diff = np.log(np.abs(evidence_analytic - np.exp(ln_evidence)))
            print("|evidence_analytic - evidence| / evidence = {}\n"
                  .format(np.exp(diff - ln_evidence)))

            if verbose: print("evidence_inv_analytic = {}"
                .format(1.0/evidence_analytic))
            if verbose: print("evidence_inv = {}"
                .format(ev.evidence_inv))
            if verbose: print("evidence_inv_std = {}"
                .format(np.sqrt(ev.evidence_inv_var)))
            if verbose: print("evidence_inv_std / evidence_inv = {}"
                .format(np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))

            if verbose: print(
                "|evidence_inv_analytic - evidence_inv| / evidence_inv = {}"
                .format(np.abs(1.0 / evidence_analytic - ev.evidence_inv) 
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
            if verbose: print("nsamples_eff_per_chain = \n{}\n"
                .format(ev.nsamples_eff_per_chain))

            # Create corner/triangle plot.
            if plot_corner:
                
                labels = [r'$\mu$', r'$\tau$']
                utils.plot_corner(samples.reshape((-1, ndim)), labels)
                if savefigs:
                    plt.savefig('./plots/normgamma_corner_tau' + 
                                str(tau_prior) +
                                '.pdf',
                                bbox_inches='tight')
                                
                labels = [r'\mu', r'\tau']
                utils.plot_getdist(samples.reshape((-1, ndim)), labels)
                if savefigs:
                    plt.savefig('./plots/normgamma_getdist_tau' + 
                                str(tau_prior) +
                                '.pdf',
                                bbox_inches='tight')
                
                plt.show(block=False)  
                created_plots = True
                
    # Display summary results.    
    print("tau_prior | ln_evidence_analytic | ln_evidence =")
    print("{}".format(summary[:,:-1]))        
                
    # Plot evidence values for different tau priors.
    if plot_comparison:
        created_plots = True        
            
        matplotlib.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots()
        ax.plot(np.array([1E-5, 1E1]), np.ones(2), 'r', linewidth=2)
        ax.set_xlim([1E-5, 1E1])
        ax.set_ylim([0.990, 1.010])
        ax.set_xscale("log")
        ax.set_xlabel(r"Prior size ($\tau_0$)")
        ax.set_ylabel(r"Relative accuracy ($z_{\rm estimated}/z_{\rm analytic}$)")        
        ax.errorbar(tau_array, np.exp(summary[:,2])/np.exp(summary[:,1]), 
            yerr=np.exp(summary[:,3])/np.exp(summary[:,1]), 
            fmt='b.', capsize=4, capthick=2, elinewidth=2)
        if savefigs:
            plt.savefig('./plots/normgamma_comparison.pdf',
                        bbox_inches='tight')                        
        plt.show(block=False)   
           
    if created_plots:
        input("\nPress Enter to continue...")


if __name__ == '__main__':
    
    # Define parameters.
    ndim = 2 # Only 2 dimensional case supported.
    nchains = 200
    samples_per_chain = 1500
    nburn = 500
    np.random.seed(1)
    
    # Run example.
    samples = run_example(ndim, nchains, samples_per_chain, nburn, 
                          plot_corner=False, plot_comparison=True, verbose=False)

