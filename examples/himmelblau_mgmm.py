import numpy as np
import sys
import emcee
import time 
import matplotlib.pyplot as plt
from functools import partial
sys.path.append(".")
import harmonic as hm
sys.path.append("examples")
import utils


def ln_prior_uniform(x, xmin=-6.0, xmax=6.0, ymin=-6.0, ymax=6.0):
    """Compute log_e of uniform prior.

    Args:

        x: Position at which to evaluate prior.

        xmin: Uniform prior minimum x edge (first dimension).

        xmax: Uniform prior maximum x edge (first dimension).

        ymin: Uniform prior minimum y edge (second dimension).

        ymax: Uniform prior maximum y edge (second dimension).

    Returns:

        double: Value of prior at specified point.

    """
        
    if x[0] >= xmin and x[0] <= xmax and x[1] >= ymin and x[1] <= ymax:        
        return 1.0 / ( (xmax - xmin) * (ymax - ymin) )
    else:
        return 0.0
        

def ln_likelihood(x):
    """Compute log_e of likelihood defined by Rastrigin function.

    Args:

        x: Position at which to evaluate likelihood.

    Returns:

        double: Value of Rastrigin at specified point.

    """
    
    f = (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2

    return -f


def ln_posterior(x, ln_prior):
    """Compute log_e of posterior.

    Args:

        x: Position at which to evaluate posterior.

        ln_prior: Prior function.

    Returns:

        double: Posterior at specified point.

    """
    
    ln_L = ln_likelihood(x)

    if not np.isfinite(ln_L):
        return -np.inf
    else:
        return ln_prior(x) + ln_L



def run_example(ndim=2, nchains=100, samples_per_chain=1000, 
                nburn=500, plot_corner=False, plot_surface=False):
    """Run Himmelblau example.

    Args:

        ndim: Dimension.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        nburn: Number of burn in samples for each chain.

        plot_corner: Plot marginalised distributions if true.

        plot_surface: Plot surface and samples if true.

    """
         
    if ndim != 2:
        raise ValueError("Only ndim=2 is supported (ndim={} specified)"
            .format(ndim))

    #===========================================================================
    # Configure Parameters.
    #===========================================================================
    """
    Configure machine learning parameters
    """
    savefigs = True
    # Set parameters
    nfold=2
    # KDE
    domains_KDE = []
    hyper_parameters_KDE = [0.01] # need double [] for corss-val
    hm.logs.debug_log('Hyper-parameters KDE= {}'.format(hyper_parameters_KDE))
    # MGMM
    domains_MGMM = [np.array([1E-1,6E0])]
    hyper_parameters_MGMM=[4, 1E-8, 2, 10, 10] # need double [] for corss-val
    hm.logs.debug_log('Hyper-parameters MGMM= {}'.format(hyper_parameters_MGMM))
    """
    Set prior parameters.
    """
    use_uniform_prior = True
    if use_uniform_prior:        
        xmin = -6.0
        xmax = 6.0
        ymin = -6.0
        ymax = 6.0
        hm.logs.debug_log('xmin, xmax, ymin, ymax = {}, {}, {}, {}'
            .format(xmin, xmax, ymin, ymax))   
        ln_prior = partial(ln_prior_uniform, \
                           xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    
    # Start timer.
    clock = time.process_time()

    #===========================================================================
    # Begin multiple realisations of estimator
    #===========================================================================
    """
    Set up and run multiple simulations
    """
    # Initialise empty lists for collation
    kurtosis_list=[]
    eff_samp_list=[]
    sqrt_list=[]


    n_realisations = 100 # change to 100 when ready
    evidence_inv_summary = np.zeros((n_realisations,3))
    for i_realisation in range(n_realisations):
        if n_realisations > 1:
            hm.logs.info_log('Realisation number = {}/{}'
                .format(i_realisation+1, n_realisations)) #n_real iterator on 1
        
        #=======================================================================
        # Run Emcee to recover posterior samples
        #=======================================================================
        hm.logs.info_log('Run sampling...')
        """
        Feed emcee the ln_posterior function, starting positions and recover 
        chains.
        """
        pos = np.random.rand(ndim * nchains).reshape((nchains, ndim))*10.0-5.0
        sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, \
                                        args=[ln_prior])
        rstate = np.random.get_state()
        sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
        samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
        lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

        #=======================================================================
        # Configure emcee chains for harmonic
        #=======================================================================
        hm.logs.info_log('Configure chains...')
        """
        Configure chains for the cross validation stage.
        """
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)
        chains_train, chains_test = hm.utils.split_data(chains, 
                                                        training_proportion=0.5)

        #=======================================================================
        # Perform cross-validation
        #=======================================================================
        hm.logs.info_log('Perform cross-validation...')
        """
        There are several different machine learning models. Cross-validation
        allows the software to select the optimal model and the optimal model 
        hyper-parameters for a given situation.
        """
        # validation_variances_KDE = \
        #     hm.utils.cross_validation(
        #             chains_train, \
        #             domains_KDE, \
        #             hyper_parameters_KDE, \
        #             nfold=nfold, \
        #             modelClass=hm.model.KernelDensityEstimate, \
        #             seed=0)
        # best_hyper_param_ind_KDE = np.argmin(validation_variances_KDE)
        # best_hyper_param_KDE = hyper_parameters_KDE[best_hyper_param_ind_KDE]
        # best_var_KDE = validation_variances_KDE[best_hyper_param_ind_KDE]

        # validation_variances_MGMM = \
        #     hm.utils.cross_validation(
        #             chains_train, \
        #             domains_MGMM, \
        #             hyper_parameters_MGMM, \
        #             nfold=nfold, \
        #             modelClass=hm.model.ModifiedGaussianMixtureModel, \
        #             seed=0)
        # best_hyper_param_ind_MGMM = np.argmin(validation_variances_MGMM)
        # best_hyper_param_MGMM = hyper_parameters_MGMM[best_hyper_param_ind_MGMM]
        # best_var_MGMM = validation_variances_MGMM[best_hyper_param_ind_MGMM]

        #=======================================================================
        # Fit optimal model hyper-parameters
        #=======================================================================
        hm.logs.info_log('Fit model...')
        """
        Fit model by selecing the configuration of hyper-parameters which 
        minimises the validation variances.
        """
        # if best_var_MGMM < best_var_KDE:                        
        #     model = hm.model.ModifiedGaussianMixtureModel(ndim, domains_MGMM, 
        #                              hyper_parameters=best_hyper_param_MGMM)
        #     hm.logs.info_log('Using MGMM model!')
  
        # else:                       
        #     model = hm.model.KernelDensityEstimate(ndim, domains_KDE, 
        #                                 hyper_parameters=best_hyper_param_KDE)
        #     hm.logs.info_log('Using KDE model!')

        # model = hm.model.KernelDensityEstimate(ndim, domains_KDE, 
        #                                 hyper_parameters=hyper_parameters_KDE)
        model = hm.model.ModifiedGaussianMixtureModel(ndim, domains_MGMM, 
                                     hyper_parameters=hyper_parameters_MGMM)
        fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
        #=======================================================================
        # Computing evidence using learnt model and emcee chains
        #=======================================================================
        hm.logs.info_log('Compute evidence...')
        """
        Instantiates the evidence class with a given model. Adds some chains and 
        computes the log-space evidence (marginal likelihood).
        """
        ev = hm.Evidence(chains_test.nchains, model)    
        ev.add_chains(chains_test)
        ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
        
        # Compute analytic evidence.
        if ndim == 2:
            hm.logs.debug_log('Compute evidence by numerical integration...')
            ln_posterior_func = partial(ln_posterior, ln_prior=ln_prior)
            ln_posterior_grid, x_grid, y_grid = \
                utils.eval_func_on_grid(ln_posterior_func, 
                                        xmin=-6.0, xmax=6.0, 
                                        ymin=-6.0, ymax=6.0, 
                                        nx=1000, ny=1000)
            dx = x_grid[0,1] - x_grid[0,0]
            dy = y_grid[1,0] - y_grid[0,0]
            evidence_numerical_integration = \
                                     np.sum(np.exp(ln_posterior_grid)) * dx * dy
     
        # ======================================================================
        # Display evidence computation results.
        # ======================================================================
        hm.logs.debug_log('Evidence: numerical = {}, estimate = {}'
            .format(evidence_numerical_integration, np.exp(ln_evidence)))
        hm.logs.debug_log('Evidence: std = {}, std / estimate = {}'
            .format( np.exp(ln_evidence_std), \
                     np.exp(ln_evidence_std - ln_evidence)))
        diff = np.log( np.abs(evidence_numerical_integration \
                     - np.exp(ln_evidence)))
        hm.logs.info_log('Evidence: |numerical - estimate| / estimate = {}'
            .format(np.exp(diff - ln_evidence)))

        # ======================================================================
        # Display inverse evidence computation results.
        # ======================================================================
        hm.logs.debug_log('Inv Evidence: numerical = {}, estimate = {}'
            .format(1.0/evidence_numerical_integration, ev.evidence_inv))
        hm.logs.debug_log('Inv Evidence: std = {}, std / estimate = {}'
            .format(np.sqrt(ev.evidence_inv_var), \
                    np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
        hm.logs.debug_log('Inv Evidence: kurtosis = {}, sqrt( 2 / ( n_eff - 1 ) ) = {}'
            .format(ev.kurtosis, np.sqrt( 2.0 / (ev.n_eff-1) )))    
        hm.logs.debug_log('Inv Evidence: sqrt( var(var) )/ var = {}'
            .format(np.sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var))    
        hm.logs.info_log('Inv Evidence: |numerical - estimate| / estimate = {}'
            .format(np.abs(1.0 / evidence_numerical_integration - \
                ev.evidence_inv) / ev.evidence_inv))

        # ======================================================================
        # Display more technical details for ln evidence.
        # ======================================================================
        hm.logs.debug_log('---------------------------------')
        hm.logs.debug_log('Technical Details')
        hm.logs.debug_log('---------------------------------')
        hm.logs.debug_log('lnargmax = {}, lnargmin = {}'
            .format(ev.lnargmax, ev.lnargmin))
        hm.logs.debug_log('lnprobmax = {}, lnprobmin = {}'
            .format(ev.lnprobmax, ev.lnprobmin))
        hm.logs.debug_log('lnpredictmax = {}, lnpredictmin = {}'
            .format(ev.lnpredictmax, ev.lnpredictmin))
        hm.logs.debug_log('---------------------------------')
        hm.logs.debug_log('shift = {}, shift setting = {}'
            .format(ev.shift_value, ev.shift))
        # hm.logs.debug_log('shift = {}, max shift = {}' # need to ask what to do about these
        #     .format(ev.mean_shift, ev.max_shift))
        hm.logs.debug_log('running sum total = {}'
            .format(sum(ev.running_sum)))
        hm.logs.debug_log('running sum = \n{}'
            .format(ev.running_sum))
        hm.logs.debug_log('nsamples per chain = \n{}'
            .format(ev.nsamples_per_chain))
        hm.logs.debug_log('nsamples eff per chain = \n{}'
            .format(ev.nsamples_eff_per_chain))
        hm.logs.debug_log('===============================')


        # Create corner/triangle plot.
        created_plots = False
        if plot_corner and i_realisation == 0:
            
            utils.plot_corner(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('examples/plots/himmelblau_corner.png',
                            bbox_inches='tight')
            
            utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('examples/plots/himmelblau_getdist.png',
                            bbox_inches='tight')
            
            plt.show(block=False)  
            created_plots = True
                
        # In 2D case, plot surface/image and samples.    
        if plot_surface and ndim == 2 and i_realisation == 0:
            
            # Plot ln_posterior surface.
            i_chain = 0
            ax = utils.plot_surface(ln_posterior_grid, x_grid, y_grid, 
                                    samples[i_chain,:,:].reshape((-1, ndim)), 
                                    lnprob[i_chain,:].reshape((-1, 1)),
                                    contour_z_offset=-850)              
            ax.set_zlabel(r'$\log \mathcal{L}$')        
            if savefigs:
                plt.savefig('examples/plots/himmelblau_lnposterior_surface.png',
                            bbox_inches='tight')
            
            # Plot posterior image.
            ax = utils.plot_image(np.exp(ln_posterior_grid), x_grid, y_grid, 
                                  samples.reshape((-1,ndim)),
                                  colorbar_label=r'$\mathcal{L}$',
                                  plot_contour=True)
            if savefigs:
                plt.savefig('examples/plots/himmelblau_posterior_image.png',
                            bbox_inches='tight')

            # Evaluate model on grid.
            model_grid, x_grid, y_grid = \
                utils.eval_func_on_grid(model.predict, 
                                        xmin=xmin, xmax=xmax, 
                                        ymin=ymin, ymax=ymax, 
                                        nx=1000, ny=1000)
            
            # Plot model.
            ax = utils.plot_image(model_grid, x_grid, y_grid, 
                                  colorbar_label=r'$\log \varphi$') 
            if savefigs:
                plt.savefig('examples/plots/himmelblau_model_image.png',
                            bbox_inches='tight')
            
            # Plot exponential of model.
            ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                                  colorbar_label=r'$\varphi$')       
            if savefigs:
                plt.savefig('examples/plots/himmelblau_modelexp_image.png',
                            bbox_inches='tight')

            plt.show(block=False)  
            created_plots = True

        # Save out realisations for voilin plot.
        evidence_inv_summary[i_realisation,0] = ev.evidence_inv
        evidence_inv_summary[i_realisation,1] = ev.evidence_inv_var
        evidence_inv_summary[i_realisation,2] = ev.evidence_inv_var_var

        #collation time
        kurtosis_list.append(ev.kurtosis)
        eff_samp_list.append(np.mean(ev.nsamples_eff_per_chain))
        sqrt_list.append(np.sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var)

    kurtosis_mean=np.mean(kurtosis_list)
    kurtosis_std=np.std(kurtosis_list)
    eff_samp_mean=np.mean(eff_samp_list)
    eff_samp_std=np.std(eff_samp_list)
    sqrt_mean=np.mean(sqrt_list)
    sqrt_std=np.std(sqrt_list)

    hm.logs.debug_log('collated kurtosis = {} ± {}'.format(kurtosis_mean, kurtosis_std))
    hm.logs.debug_log('collated eff_samp = {} ± {}'.format(eff_samp_mean, eff_samp_std))
    hm.logs.debug_log('collated sqrt_var = {} ± {}'.format(sqrt_mean, sqrt_std))
        

    #===========================================================================
    # End Timer.
    clock = time.process_time() - clock
    hm.logs.info_log('Execution time = {}s'.format(clock))

    #===========================================================================
    
    # Save collations

    np.savetxt("examples/data/kurtosis_mean_mgmm",
                   np.array([kurtosis_mean]))
    np.savetxt("examples/data/kurtosis_std_mgmm",
                   np.array([kurtosis_std]))
    np.savetxt("examples/data/eff_samp_mean_mgmm",
                   np.array([eff_samp_mean]))
    np.savetxt("examples/data/eff_samp_std_mgmm",
                   np.array([eff_samp_std]))
    np.savetxt("examples/data/sqrt_mean_mgmm",
                   np.array([sqrt_mean]))
    np.savetxt("examples/data/sqrt_std_mgmm",
                   np.array([sqrt_std]))
    
    
    
    # Save out realisations of statistics for analysis.
    if n_realisations > 1:
        np.savetxt("examples/data/himmelblau_evidence_inv" +
                   "_realisations_mgmm.dat",
                   evidence_inv_summary)
        evidence_inv_analytic_summary = np.zeros(1)
        evidence_inv_analytic_summary[0] = 1.0 / evidence_numerical_integration
        np.savetxt("examples/data/himmelblau_evidence_inv" +
                   "_analytic_mgmm.dat",
                   evidence_inv_analytic_summary)

    if created_plots:
        input("\nPress Enter to continue...")
    
    return samples


if __name__ == '__main__':

    # Setup logging config.
    hm.logs.setup_logging()
    
    # Define parameters.
    ndim = 2
    nchains = 200
    samples_per_chain = 5000
    nburn = 2000
    np.random.seed(20)
    
    hm.logs.info_log('Himmelblau example')

    hm.logs.debug_log('-- Selected Parameters --')
    
    hm.logs.debug_log('Dimensionality = {}'.format(ndim))
    hm.logs.debug_log('Number of chains = {}'.format(nchains))
    hm.logs.debug_log('Samples per chain = {}'.format(samples_per_chain))
    hm.logs.debug_log('Burn in = {}'.format(nburn))
    
    hm.logs.debug_log('-------------------------')  

    # Run example.
    samples = run_example(ndim, nchains, samples_per_chain, nburn, 
                          plot_corner=True, plot_surface=True)