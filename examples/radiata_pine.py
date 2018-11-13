import numpy as np
import sys
import emcee
import scipy.special as sp
import time 
import matplotlib.pyplot as plt
from functools import partial
sys.path.append(".")
import harmonic as hm
sys.path.append("examples")
import utils

# Setup Logging config
hm.logs.setup_logging()



def ln_likelihood(y, x, n, alpha, beta, tau):
    
    ln_like = 0.5 * n * np.log(tau)
    ln_like -= 0.5 * n * np.log(2.0 * np.pi)
    
    s = np.sum((y - alpha - beta * x)**2)
    
    ln_like -= 0.5 * tau * s
    
    return ln_like
    
    
def ln_likelihood_check(y, x, n, alpha, beta, tau):
    
    ln_like = 0.5 * n * np.log(tau)
    ln_like -= 0.5 * n * np.log(2.0 * np.pi)
    
    s = 0.0
    for i in range(n):
        s += (y[i,0] - alpha - beta * x[i,0])**2
        
    ln_like -= 0.5 * tau * s
    
    return ln_like
    
    
def ln_prior_alpha(alpha, tau, mu_0, r_0):
    """
        mu_0 here is scalar
    """
    
    ln_pr_alpha = 0.5 * np.log(tau)
    ln_pr_alpha += 0.5 * np.log(r_0)
    ln_pr_alpha -= 0.5 * np.log(2.0 * np.pi)
    ln_pr_alpha -= 0.5 * tau * r_0 * (alpha - mu_0)**2
    
    return ln_pr_alpha


def ln_prior_tau(tau, a_0, b_0):
    
    if tau < 0:
        return -np.inf
    
    ln_pr_tau = a_0 * np.log(b_0)
    ln_pr_tau += (a_0 - 1.0) * np.log(tau)
    ln_pr_tau -= b_0 * tau
    ln_pr_tau -= sp.gammaln(a_0)

    return ln_pr_tau


def ln_prior_separated(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    
    ln_pr = ln_prior_alpha(alpha, tau, mu_0[0,0], r_0)
    ln_pr += ln_prior_alpha(beta, tau, mu_0[1,0], s_0)
    ln_pr += ln_prior_tau(tau, a_0, b_0)
        
    return ln_pr


def ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    
    if tau < 0:
        return -np.inf

    ln_pr = a_0 * np.log(b_0)    
    ln_pr += a_0 * np.log(tau)    
    ln_pr -= b_0 * tau    
    ln_pr -= np.log(2.0 * np.pi)
    ln_pr -= sp.gammaln(a_0)
    ln_pr += 0.5 * np.log(r_0)
    ln_pr += 0.5 * np.log(s_0)
    ln_pr -= 0.5 * tau \
             * (r_0 * (alpha - mu_0[0,0])**2 + s_0 * (beta - mu_0[1,0])**2)
    
    return ln_pr


def ln_prior(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):

    return ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)
        

def ln_posterior(theta, y, x, n, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of posterior.
    
    Args: 
        theta: Position (alpha, beta, tau) at which to evaluate posterior.
        ... 
        
    Returns:
        double: Value of log_e posterior at specified (alpha, beta, tau) point.
    """
    
    alpha, beta, tau = theta
    # print("alpha, beta, tau = ({}, {}, {})".format(alpha, beta, tau))

    ln_pr = ln_prior(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)
    # print("ln_pr = {}".format(ln_pr))
    
    if not np.isfinite(ln_pr):
        return -np.inf

    ln_L = ln_likelihood(y, x, n, alpha, beta, tau)    
    # print("ln_L = {}\n".format(ln_L))
    
    return  ln_L + ln_pr
    
    
    
    
    
    
    

def run_example(ndim=3, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False,
                plot_comparison=False):
    
    hm.logs.low_log('---------------------------------')
    hm.logs.high_log('Radiata Pine example')
    hm.logs.high_log('Dimensionality = {}'.format(ndim))
    hm.logs.low_log('---------------------------------')

    if ndim != 3:
        raise ValueError("Only ndim=3 is supported (ndim={} specified)"
            .format(ndim))
    
        
        
    # Set general parameters.    
    savefigs = True
    
    nfold = 3
    training_proportion = 0.25
    hyper_parameters_MGMM = [[1, 1E-8, 0.1, 6, 10],\
            [2, 1E-8, 0.5, 6, 10]]#, [3,1E-8,2.0,10,10]]
    hyper_parameters_sphere = [None]
    domains_sphere = [np.array([1E-1,5E0])]
    domains_MGMM = [np.array([1E-1,5E0])]
    
    
    # training_proportion = 0.50
    # nfold = 2
    # nhyper = 2
    # step = -2    
    # domain = []
    # # hyper_parameters = [[10**(R)] for R in range(-nhyper+step,step)]
    # hyper_parameters = [[10**(R)] for R in range(-3,-1,1)]
    # if verbose: print("hyper_parameters = {}".format(hyper_parameters))
    
    
    #=========================================================
    # Set-up Priors
    #=========================================================  
    # Define prior variables
    mu_0 = np.array([[3000.0], [185.0]])    
    r_0 = 0.06
    s_0 = 6.0
    a_0 = 3.0
    b_0 = 2.0 * 300**2
    
    #=========================================================
    # Load Radiata Pine data.
    #=========================================================
    # Imports data file
    data = np.loadtxt('./data/RadiataPine.dat')
    id = data[:,0]
    y = data[:,1]
    x = data[:,2]
    z = data[:,3]
    n = len(x)

    # Ensure column vectors
    y = y.reshape(n,1)
    x = x.reshape(n,1)
    z = z.reshape(n,1)

    # Remove means from covariates.
    x = x - np.mean(x)
    z = z - np.mean(z)

    # Set up and run sampler.
    tau_prior_mean = a_0 / b_0
    tau_prior_std = np.sqrt(a_0) / b_0

    #=========================================================
    # Compute random positions to draw from for emcee sampler.
    #=========================================================
    pos_alpha = mu_0[0,0] + 1.0 / np.sqrt(tau_prior_mean * r_0) * np.random.randn(nchains)  
    pos_beta = mu_0[1,0] + 1.0 / np.sqrt(tau_prior_mean * s_0) * np.random.randn(nchains)    
    # pos_tau = tau_prior_mean + tau_prior_std * np.random.randn(nchains)            
    pos_tau = tau_prior_mean + 2.0 * tau_prior_std * 2.0 * (np.random.rand(nchains) - 0.5)  # avoid negative tau
        
    pos = np.c_[pos_alpha, pos_beta, pos_tau]
           
           
    # for i in range(nchains):
    #     ln_p = ln_posterior(pos[i], y, x, n, mu_0, r_0, s_0, a_0, b_0)
    #     print("pos[i] = {}".format(pos[i]))
    #     print("ln_p = {}\n".format(ln_p))
    # 
    # return 
    # 
    
    # Start timer.
    clock = time.clock()
    
    hm.logs.high_log('Run sampling...')
    
    # #=========================================================
    # # Temporary position for testing
    # #=========================================================
    # pos = np.random.rand(ndim * nchains).reshape((nchains, ndim)) * 0.1
    
    #=========================================================
    # Run Emcee to recover posterior sampels 
    #=========================================================
    sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, \
        # args=(y, x, n, mu_0, r_0, s_0, a_0, b_0))
        args=(y, z, n, mu_0, r_0, s_0, a_0, b_0))
    rstate = np.random.get_state()
    sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
    samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
    lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

    # print("samples = {}".format(samples))
    # print("lnprob = {}".format(lnprob)) 
    # print("sampler.chain = ".format(sampler.chain))
    
    
    #=========================================================
    # Configure emcee chains for harmonic
    #=========================================================
    hm.logs.low_log('---------------------------------')
    hm.logs.high_log('Calculate evidence using harmonic...')

    #=========================================================
    # Configure emcee chains for harmonic
    #=========================================================
    hm.logs.low_log('---------------------------------')
    hm.logs.high_log('Configuring chains...')
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, lnprob)
    chains_train, chains_test = hm.utils.split_data(chains, \
        training_proportion=training_proportion)
        
    #=========================================================
    # Perform cross-validation
    #=========================================================
    hm.logs.low_log('---------------------------------')
    hm.logs.high_log('Perform cross-validation...')
    
    # validation_variances_MGMM = \
    #     hm.utils.cross_validation(chains_train, 
    #         domains_MGMM, \
    #         hyper_parameters_MGMM, \
    #         nfold=nfold, 
    #         modelClass=hm.model.ModifiedGaussianMixtureModel, \
    #         verbose=verbose, seed=0)                
    # if verbose: print("validation_variances_MGMM = {}"
    #     .format(validation_variances_MGMM))
    # best_hyper_param_MGMM_ind = np.argmin(validation_variances_MGMM)
    # best_hyper_param_MGMM = \
    #     hyper_parameters_MGMM[best_hyper_param_MGMM_ind]
    # 
    # validation_variances_sphere = \
    #     hm.utils.cross_validation(chains_train, 
    #         domains_sphere, \
    #         hyper_parameters_sphere, nfold=nfold, 
    #         modelClass=hm.model.HyperSphere, 
    #         verbose=verbose, seed=0)
    # if verbose: print("validation_variances_sphere = {}"
    #     .format(validation_variances_sphere))
    # best_hyper_param_sphere_ind = np.argmin(validation_variances_sphere)
    # best_hyper_param_sphere = \
    #     hyper_parameters_sphere[best_hyper_param_sphere_ind]

    # Perform cross-validation.
    # print("Perform cross-validation...")
    # validation_variances = \
    #     hm.utils.cross_validation(chains_train, \
    #                               domain, \
    #                               hyper_parameters, \
    #                               nfold=nfold, \
    #                               modelClass=hm.model.KernelDensityEstimate, \
    #                               verbose=verbose, \
    #                               seed=0)
    # if verbose: print("validation_variances = {}".format(validation_variances))
    # best_hyper_param_ind = np.argmin(validation_variances)
    # best_hyper_param = hyper_parameters[best_hyper_param_ind]
    # if verbose: print("best_hyper_param = {}".format(best_hyper_param))


    #=========================================================
    # Fit learnt model for container function 
    #=========================================================
    hm.logs.low_log('---------------------------------')
    hm.logs.high_log('Fit model...')
    # best_var_MGMM = \
    #     validation_variances_MGMM[best_hyper_param_MGMM_ind]
    # best_var_sphere = \
    #     validation_variances_sphere[best_hyper_param_sphere_ind]
    # if best_var_MGMM < best_var_sphere:            
    #     print("Using MGMM with hyper_parameters = {}"
    #         .format(best_hyper_param_MGMM))                
    #     model = hm.model.ModifiedGaussianMixtureModel(ndim, \
    #         domains_MGMM, hyper_parameters=best_hyper_param_MGMM)
    #     model.verbose=False
    # else:
    hm.logs.high_log('Using HyperSphere')
    # model = hm.model.HyperSphere(ndim, domains_sphere, hyper_parameters=best_hyper_param_sphere)
    model = hm.model.HyperSphere(ndim, domains_sphere, hyper_parameters=None)            
        
    fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
    hm.logs.low_log('fit_success = {}'.format(fit_success))    
    
    model.set_R(model.R * 0.5) # conservative reduction in R.
    # model.set_R(0.5)
    hm.logs.low_log('model.R = {}'.format(model.R))
    

    # model = hm.model.KernelDensityEstimate(ndim, 
    #                                        domain, 
    #                                        hyper_parameters=best_hyper_param)
    # fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
    # if verbose: print("fit_success = {}".format(fit_success))   

    #=========================================================
    # Computing evidence using learnt model and emcee chains
    #=========================================================
    hm.logs.low_log('---------------------------------')
    # Use chains and model to compute evidence.
    hm.logs.high_log('Compute evidence...')
    ev = hm.Evidence(chains_test.nchains, model)
    ev.add_chains(chains_test)
    ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
    #
    # # Compute analytic evidence.
    # ln_evidence_analytic = ln_analytic_evidence(x_mean, \
    #     x_std, x_n, prior_params)
    # evidence_analytic = np.exp(ln_evidence_analytic)    
        
        
    clock = time.clock() - clock
    hm.logs.high_log('execution_time = {}s'.format(clock))
    
    
    #=========================================================
    # Display logarithmic evidence results 
    #=========================================================
    hm.logs.low_log('---------------------------------')
    # print("ln_evidence_analytic = {}"
    #     .format(ln_evidence_analytic))
    hm.logs.low_log('ln_evidence = {}, -np.log(ev.evidence_inv) = {}'
        .format(ln_evidence, -np.log(ev.evidence_inv)))            
    # diff = np.abs(ln_evidence_analytic - ln_evidence)
    # print("|ln_evidence_analytic - ln_evidence| / ln_evidence = {}\n"
    #       .format(diff/ln_evidence))

    #=========================================================
    # Display evidence results 
    #=========================================================
    hm.logs.low_log('---------------------------------')
    # print("evidence_analytic = {}"
    #     .format(evidence_analytic))
    hm.logs.low_log('evidence = {}'
        .format(np.exp(ln_evidence)))
    hm.logs.low_log('evidence_std = {}, evidence_std / evidence = {}'
        .format(np.exp(ln_evidence_std), np.exp(ln_evidence_std - ln_evidence)))
    # diff = np.log(np.abs(evidence_analytic - np.exp(ln_evidence)))
    # print("|evidence_analytic - evidence| / evidence = {}\n"
    #       .format(np.exp(diff - ln_evidence)))

    #=========================================================
    # Display inverse evidence results 
    #=========================================================
    hm.logs.low_log('---------------------------------')
    # if verbose: print("evidence_inv_analytic = {}"
    #     .format(1.0/evidence_analytic))
    hm.logs.low_log('evidence_inv = {}'
        .format(ev.evidence_inv))
    hm.logs.low_log('evidence_inv_std = {}, evidence_inv_std / evidence_inv = {}'
        .format(np.sqrt(ev.evidence_inv_var), np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
    hm.logs.low_log('kurtosis = {}, sqrt( 2 / ( n_eff - 1 ) ) = {}'
        .format(ev.kurtosis, np.sqrt(2.0/(ev.n_eff-1))))    
    hm.logs.low_log('sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var = {}'
        .format(np.sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var))
    # if verbose: print(
    #     "|evidence_inv_analytic - evidence_inv| / evidence_inv = {}"
    #     .format(np.abs(1.0 / evidence_analytic - ev.evidence_inv) 
    #             / ev.evidence_inv))

    #=========================================================
    # Display more technical details
    #=========================================================
    hm.logs.low_log('---------------------------------')
    hm.logs.low_log('lnargmax = {}, lnargmin = {}'
        .format(ev.lnargmax, ev.lnargmin))
    hm.logs.low_log('lnprobmax = {}, lnprobmin = {}'
        .format(ev.lnprobmax, ev.lnprobmin))
    hm.logs.low_log('lnpredictmax = {}, lnpredictmin = {}'
        .format(ev.lnpredictmax, ev.lnpredictmin))
    hm.logs.low_log('---------------------------------')
    hm.logs.low_log('mean shift = {}, running sum total = {}'
        .format(ev.mean_shift, sum(ev.running_sum)))
    hm.logs.low_log('running sum = \n{}'
        .format(ev.running_sum))
    hm.logs.low_log('nsamples per chain = \n{}'
        .format(ev.nsamples_per_chain))
    hm.logs.low_log('nsamples eff per chain = \n{}'
        .format(ev.nsamples_eff_per_chain))
    hm.logs.low_log('===============================')
    
    

    # Create corner/triangle plot.
    created_plots = False
    if plot_corner:
        
        utils.plot_corner(samples.reshape((-1, ndim)))
        if savefigs:
            plt.savefig('./plots/radiatapine_corner.png',
                        bbox_inches='tight')
        
        utils.plot_getdist(samples.reshape((-1, ndim)))
        if savefigs:
            plt.savefig('./plots/radiatapine_getdist.png',
                        bbox_inches='tight')
        
        plt.show(block=False)  
        created_plots = True



    # TODO
    
    
    # plot model, take 2D slices starting from mean or MAP estimate in other dimension
    
    
    # Evaluate model on grid.
    
    #=========================================================
    # BELOW HERE IS ESSENTIALLY RAW FROM JASON'S LAST COMMIT
    #=========================================================
    
    
    def model_predict_x0x1(x_2d):         
        # x2 = a_0 / b_0
        x2 = 1.4E-5
        x = np.append(x_2d, [x2])
        # print("x01x1: x = {}".format(x))
        return model.predict(x)
        
    model_grid, x_grid, y_grid = \
        utils.eval_func_on_grid(model_predict_x0x1, 
                                xmin=2900.0, xmax=3100.0, 
                                ymin=185.0-30.0, ymax=185.0+30.0,
                                nx=1000, ny=1000)
                                                                
    # Plot model.
    ax = utils.plot_image(model_grid, x_grid, y_grid, 
                          colorbar_label=r'$\log \varphi$')   
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')   
    #plt.axis('equal')
    
    if savefigs:
        plt.savefig('./plots/radiatapine_model_x0x1_image.png',
                    bbox_inches='tight')

    # Plot exponential of model.
    ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                          colorbar_label=r'$\varphi$')    
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')   
    #plt.axis('equal')    
    if savefigs:
        plt.savefig('./plots/radiatapine_modelexp_x0x1_image.png',
                    bbox_inches='tight')




    def model_predict_x1x2(x_2d): 
        x0 = 3000.0
        x = np.append([x0], x_2d)
        # print("x1x2: x = {}".format(x))
        return model.predict(x)
        
    model_grid, x_grid, y_grid = \
        utils.eval_func_on_grid(model_predict_x1x2, 
                                xmin=185.0-30.0, xmax=185.0+30.0, 
                                ymin=a_0 / b_0 - 0.5E-5, ymax=a_0 / b_0 + 0.5E-5, 
                                nx=1000, ny=1000)
                                                                
    # Plot model.
    ax = utils.plot_image(model_grid, x_grid, y_grid, 
                          colorbar_label=r'$\log \varphi$')   
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')   
    #plt.axis('equal')
    
    if savefigs:
        plt.savefig('./plots/radiatapine_model_x1x2_image.png',
                    bbox_inches='tight')

    # Plot exponential of model.
    ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                          colorbar_label=r'$\varphi$')    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')   
    #plt.axis('equal')    
    if savefigs:
        plt.savefig('./plots/radiatapine_modelexp_x1x2_image.png',
                    bbox_inches='tight')






    def model_predict_x0x2(x_2d): 
        x1 = 185.0
        x = np.append(x_2d[0], [x1])
        x = np.append(x, x_2d[1])
        # print("x0x2: x = {}".format(x))
        return model.predict(x)
        
    model_grid, x_grid, y_grid = \
        utils.eval_func_on_grid(model_predict_x0x2, 
                                xmin=2900.0, xmax=3100.0, 
                                ymin=a_0 / b_0 - 0.5E-5, ymax=a_0 / b_0 + 0.5E-5, 
                                nx=1000, ny=1000)
                                                                
    # Plot model.
    ax = utils.plot_image(model_grid, x_grid, y_grid, 
                          colorbar_label=r'$\log \varphi$')   
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')   
    #plt.axis('equal')
    
    if savefigs:
        plt.savefig('./plots/radiatapine_model_x0x2_image.png',
                    bbox_inches='tight')

    # Plot exponential of model.
    ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                          colorbar_label=r'$\varphi$')    
    plt.xlabel('$x_0$')
    plt.ylabel('$x_2$')   
    #plt.axis('equal')    
    if savefigs:
        plt.savefig('./plots/radiatapine_modelexp_x0x2_image.png',
                    bbox_inches='tight')














    plt.show(block=False)  



    # Test
    alpha = 3000.0 
    beta = 185.0
    tau = a_0 / b_0
    ln_pr_combined = ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)
    ln_pr_separated = ln_prior_separated(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)
    print("ln_pr_combined = {}".format(ln_pr_combined))
    print("ln_pr_separated = {}".format(ln_pr_separated))
    

    ln_like = ln_likelihood(y, x, n, alpha, beta, tau)
    ln_like_check = ln_likelihood_check(y, x, n, alpha, beta, tau)
    
    print("ln_like = {}".format(ln_pr_combined))
    print("ln_like_check = {}".format(ln_pr_separated))
    
    
    


    if created_plots:
        input("\nPress Enter to continue...")


    


if __name__ == '__main__':
    
    # Define parameters.
    ndim = 3 # Only 3 dimensional case supported.
    nchains = 200
    # samples_per_chain = 1000000
    samples_per_chain = 5000
    nburn = 1000
    np.random.seed(3)
    
    # Run example.
    samples = run_example(ndim, nchains, samples_per_chain, nburn, 
                          plot_corner=True, plot_surface=True,
                          plot_comparison=True, 
                          verbose=True)

