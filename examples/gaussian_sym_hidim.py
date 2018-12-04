import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
import emcee
import scipy.special as sp
import time 
import matplotlib.pyplot as plt
import utils

# Setup Logging config
hm.logs.setup_logging()


def ln_analytic_evidence(ndim, cov):
    """
    Compute analytic ln_e evidence.
    Args: 
        - ndim: 
            Dimensionality of the multivariate Gaussian posterior
        - cov
            Covariance matrix dimension nxn.           
    Returns:
        - double: 
            Value of posterior at x.
    """
    ln_norm_lik = -0.5*ndim*np.log(2*np.pi)-0.5*np.log(np.linalg.det(cov))   
    #TODO: diagonal covariance (same variance in each direction)
    return -ln_norm_lik

def ln_Posterior(x, inv_cov):
    """
    Compute log_e of n dimensional multivariate gaussian 
    Args: 
        - x: 
            Position at which to evaluate prior.         
    Returns:
        - double: 
            Value of posterior at x.
    """
    # return -sum( i*i/2.0 for i in x) 
    #computes nD Gaussian where each dimension has variance = 1.
    return -np.dot(x,np.dot(inv_cov,x))/2.0   
    #TODO: avoid doing matrix multiplication here (set sigma=1)

def init_cov(ndim): 
    """
    Initialise random diagonal covariance matrix.
    Args: 
        - ndim: 
            Dimension of Gaussian.        
    Returns:
        - cov: 
            Covariance matrix of shape (ndim,ndim).
    """

    cov = np.zeros((ndim,ndim))
    diag_cov = np.ones(ndim) + np.random.randn(ndim)*0.1
    # cov[0,1] = 0.5*np.sqrt(cov[0,0]*cov[1,1])
    # cov[1,0] = 0.5*np.sqrt(cov[0,0]*cov[1,1])
    np.fill_diagonal(cov, diag_cov)
    
    return cov


def run_example(ndim=2, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False):
    """
    Run nD Gaussian example with generalized covariance matrix.
    Args: 
        - ndim: 
            Dimension of multivariate Gaussian.
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
            If True then displalnprob intermediate results.
        
    Returns:
        - None.
    """

    hm.logs.high_log('nD Guassian example')
    hm.logs.high_log('Dimensionality = {}'.format(ndim))
    hm.logs.low_log('---------------------------------')
    savefigs = True
    plot_sample = False

    # ==========================================================================
    # Initialise covariance matrix.
    # ==========================================================================
    cov = init_cov(ndim)
    inv_cov = np.linalg.inv(cov)  
    hm.logs.low_log('Covariance matrix diagonal entries = \n{}'
        .format(np.diagonal(cov)))
    hm.logs.low_log('---------------------------------')

    # ==========================================================================
    # Compute analytic log-evidence for comparison
    # ==========================================================================
    ln_rho = -ln_analytic_evidence(ndim, cov)
    hm.logs.high_log('Ln Inverse Analytic evidence = {}'.format(ln_rho))
    hm.logs.low_log('---------------------------------')

    # ==========================================================================
    #                                *** TESTING ***
    # ==========================================================================

    hyper_parameters_MGMM = [ [1, 1E-8, 0.1, 18, 10] ]#, [3,1E-8,2.0,10,10]]
    domains_MGMM = [np.array([1E0,2E1])]


    max_r_prob = np.sqrt(ndim-1)
    hm.logs.low_log('max_r_prob = {}'.format(max_r_prob))
    domains_sphere = [max_r_prob*np.array([1E0,2E1])]
    hyper_parameters_sphere = [None]


    nfold = 3
    nhyper = 2
    step = -2
    domain_KDE = []
    hyper_parameters_KDE = [[10**(R)] for R in range(-nhyper+step,step)]

    # hm.logs.low_log('Domain = {}'.format(domains))

    # ==========================================================================
    #                                *** TESTING ***
    # ==========================================================================

    # Run multiple realisations.
    n_realisations = 1
    evidence_inv_summary = np.zeros((n_realisations,3))
    # Start timer.
    clock = time.clock()
    for i_realisation in range(n_realisations):

        if n_realisations > 0:
            hm.logs.high_log('Realisation = {}/{}'
                .format(i_realisation, n_realisations))

        # ======================================================================
        # Recover a set of MCMC samples from the posterior 
        # ======================================================================

        # Set up and run sampler.
        hm.logs.high_log('Run sampling...')
        hm.logs.low_log('---------------------------------')
        pos = np.random.rand(ndim * nchains).reshape((nchains, ndim))
        hm.logs.low_log('pos.shape = {}'.format(pos.shape))
        sampler = emcee.EnsembleSampler(nchains, ndim, ln_Posterior, \
                                        args=[inv_cov])
        rstate = np.random.get_state() # Set random state to be repeatable.
        (pos, prob, state) = sampler.run_mcmc(pos, samples_per_chain, \
                                              rstate0=rstate) 
        samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
        lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

        # ======================================================================
        # Configure chains
        # ======================================================================
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)
        chains_train, chains_test = hm.utils.split_data(chains, 
                                                        training_proportion=0.5)

        # ======================================================================
        #                               *** TESTING ***
        # ======================================================================
        # hm.logs.high_log('Perform cross-validation...')

        # validation_variances_MGMM = \
        #     hm.utils.cross_validation(chains_train, \
        #                    domains_MGMM, \
        #                    hyper_parameters_MGMM, \
        #                    nfold=nfold,
        #                    modelClass=hm.model.ModifiedGaussianMixtureModel,\
        #                    verbose=False, \
        #                    seed=0)
        # hm.logs.low_log('validation_variances_MGMM = {}'
        #     .format(validation_variances_MGMM))
        # best_hyper_param_MGMM_ind = np.argmin(validation_variances_MGMM)
        # best_hyper_param_MGMM = \
                        # hyper_parameters_MGMM[best_hyper_param_MGMM_ind]



        # validation_variances_sphere = \
        #     hm.utils.cross_validation(chains_train, \
        #                    domains_sphere, \
        #                    hyper_parameters_sphere, \
        #                    nfold=nfold, \
        #                    modelClass=hm.model.HyperSphere, \
        #                    verbose=False, \
        #                    seed=0)
        # hm.logs.low_log('validation_variances_sphere = {}'
        #     .format(validation_variances_sphere))
        # best_hyper_param_sphere_ind = np.argmin(validation_variances_sphere)
        # best_hyper_param_sphere = \
                        # hyper_parameters_sphere[best_hyper_param_sphere_ind]



        # validation_variances_KDE = \
        #     hm.utils.cross_validation(chains_train, \
        #                    domain_KDE, \
        #                    hyper_parameters_KDE, \
        #                    nfold=nfold, \
        #                    modelClass=hm.model.KernelDensityEstimate, \
        #                    verbose=False, \
        #                    seed=0)
        # hm.logs.low_log('Validation_variances_KDE = {}'
        #     .format(validation_variances))
        # best_hyper_param_KDE_ind = np.argmin(validation_variances)
        # best_hyper_param_KDE = \
                        # hyper_parameters[best_hyper_param_KDE_ind]

        # ======================================================================
        #                               *** TESTING ***
        # ======================================================================

        # ======================================================================
        # Train hyper-spherical model 
        # ======================================================================
        model = hm.model.HyperSphere(ndim, domains_sphere)
        fit_success, objective = model.fit(chains_train.samples,\
                                           chains_train.ln_posterior) 
        hm.logs.low_log('Fit success = {}'.format(fit_success))    
        hm.logs.low_log('Objective = {}'.format(objective))    
        hm.logs.low_log('---------------------------------')

        # ======================================================================
        #                                *** TESTING ***
        # ======================================================================
        # # Fit model.
        # hm.logs.high_log('Fit model...')
        # best_var_MGMM = \
                # validation_variances_MGMM[best_hyper_param_MGMM_ind]
        # best_var_sphere = \
                # validation_variances_sphere[best_hyper_param_sphere_ind]
        # best_var_KDE = \
                # validation_variances_KDE[best_hyper_param_KDE_ind]

        # if best_var_MGMM < best_var_sphere:
        #     if best_var_MGMM < best_var_KDE:
        #         hm.logs.low_log('Using MGMM with hyper_parameters = {}'
        #             .format(best_hyper_param_MGMM))
        #         model = hm.model.ModifiedGaussianMixtureModel(ndim, \
        #             domains_MGMM, hyper_parameters=best_hyper_param_MGMM)
        #         model.verbose=False
        #     else:
        #         hm.logs.low_log('Using KDE with hyper_parameters = {}'
        #             .format(best_hyper_param_KDE))
        #         model = hm.model.KernelDensityEstimate(ndim, \
        #             domains_KDE, hyper_parameters=best_hyper_param_KDE)
        #         model.verbose=False
        # else:
        #     if best_var_sphere < best_var_KDE:
        #         hm.logs.low_log('Using HyperSphere')
        #         model = hm.model.HyperSphere(ndim, domains_sphere, \
        #             hyper_parameters=best_hyper_param_sphere)
        #     else:
        #         hm.logs.low_log('Using KDE with hyper_parameters = {}'
        #             .format(best_hyper_param_KDE))
        #         model = hm.model.KernelDensityEstimate(ndim, \
        #             domains_KDE, hyper_parameters=best_hyper_param_KDE)
        #         model.verbose=False

        # fit_success = model.fit(chains_train.samples,
        #                         chains_train.ln_posterior)
        # hm.logs.low_log('Fit success = {}'.format(fit_success))
        # hm.logs.low_log('---------------------------------')

        # ======================================================================
        #                                *** TESTING ***
        # ======================================================================


        # ======================================================================
        # Compute ln evidence.
        # ======================================================================
        hm.logs.high_log('Compute evidence...')
        cal_ev = hm.Evidence(chains_test.nchains, model)
        cal_ev.add_chains(chains_test)
        ln_evidence, ln_evidence_std = cal_ev.compute_ln_evidence()

        # ======================================================================
        # Display logarithmic inverse evidence computation results.
        # ======================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('Ln Inv Evidence: analytic = {}, estimate = {}'
            .format(ln_rho, np.log(cal_ev.evidence_inv)))
        hm.logs.high_log('Ln Inv Evidence: \
                          100 * |analytic - estimate| / |analytic| = {}%'
            .format(100.0 * np.abs( (np.log(cal_ev.evidence_inv) - ln_rho) \
                                                                 / ln_rho ))) 
        # ======================================================================
        # Display inverse evidence computation results.
        # ======================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('Inv Evidence: analytic = {}, estimate = {}'
            .format(np.exp(ln_rho), cal_ev.evidence_inv))
        hm.logs.low_log('Inv Evidence: std = {}, std / estimate = {}'
            .format(np.sqrt(cal_ev.evidence_inv_var), \
                    np.sqrt(cal_ev.evidence_inv_var)/cal_ev.evidence_inv))
        hm.logs.high_log("Inv Evidence: \
                          100 * |analytic - estimate| / estimate = {}%"
            .format(100.0 * np.abs( np.exp(ln_rho) - cal_ev.evidence_inv ) \
                                                   / cal_ev.evidence_inv ) )
        # ======================================================================
        # Display more technical details for ln evidence.
        # ======================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('lnargmax = {}, lnargmin = {}'
            .format(cal_ev.lnargmax, cal_ev.lnargmin))
        hm.logs.low_log('lnprobmax = {}, lnprobmin = {}'
            .format(cal_ev.lnprobmax, cal_ev.lnprobmin))
        hm.logs.low_log('lnpredictmax = {}, lnpredictmin = {}'
            .format(cal_ev.lnpredictmax, cal_ev.lnpredictmin))
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('mean shift = {}, max shift = {}'
            .format(cal_ev.mean_shift, cal_ev.max_shift))
        hm.logs.low_log('running sum total = {}'
            .format(sum(cal_ev.running_sum)))
        hm.logs.low_log('running_sum = \n{}'
            .format(cal_ev.running_sum))
        hm.logs.low_log('nsamples_per_chain = \n{}'
            .format(cal_ev.nsamples_per_chain))
        hm.logs.low_log('nsamples_eff_per_chain = \n{}'
            .format(cal_ev.nsamples_eff_per_chain))
        hm.logs.low_log('===============================')

        # ======================================================================
        # Create corner/triangle plot.
        # ======================================================================
        created_plots = False
        if plot_corner and i_realisation == 0:
            
            utils.plot_corner(samples.reshape((-1, ndim)))

            if savefigs:
                plt.savefig('./plots/nD_gaussian_corner.png',
                            bbox_inches='tight')

            plt.show(block=False)
            created_plots = True

        evidence_inv_summary[i_realisation,0] = cal_ev.evidence_inv
        evidence_inv_summary[i_realisation,1] = cal_ev.evidence_inv_var
        evidence_inv_summary[i_realisation,2] = cal_ev.evidence_inv_var_var


    clock = time.clock() - clock
    hm.logs.high_log('Execution_time = {}s'.format(clock))

    if n_realisations > 1:
        np.savetxt("./data/nD_gaussian_evidence_inv" +
                   "_realisations.dat",
                   evidence_inv_summary)
        evidence_inv_analytic_summary = np.zeros(1)
        evidence_inv_analytic_summary[0] = np.exp(ln_rho)
        np.savetxt("./data/nD_gaussian_evidence_inv" +
                   "_analytic.dat",
                   evidence_inv_analytic_summary)

    if created_plots:
        input("\nPress Enter to continue...")
    
    return samples



if __name__ == '__main__':
    
    # Define parameters.
    ndim = 180
    nchains = 360
    samples_per_chain = 24000
    nburn = 22000
    np.random.seed(10)
    
    # Run example.
    run_example(ndim, nchains, samples_per_chain, nburn, 
                plot_corner=False, plot_surface=False, verbose=False)


