import numpy as np
import sys
import emcee
import time 
import matplotlib.pyplot as plt
from functools import partial
from matplotlib import cm
sys.path.append(".")
import harmonic as hm
sys.path.append("examples")
import utils

# Setup Logging config
hm.logs.setup_logging()


def ln_analytic_evidence(ndim, cov):
    """
    Compute analytic evidence for nD Gaussian.
    Args:
        - ndim: 
            Dimension of Gaussian.
        - cov: 
            Covariance matrix.
    Returns:
        - double:
            Analytic evidence.
    """
    
    ln_norm_lik = 0.5*ndim*np.log(2*np.pi) + 0.5*np.log(np.linalg.det(cov))
    return ln_norm_lik


def ln_posterior(x, inv_cov):
    """
    Compute log_e of posterior.
    Args: 
        - x: 
            Position at which to evaluate posterior.
        - inv_cov: 
            Inverse covariance matrix.    
    Returns:
        - double: 
            Value of Gaussian at specified point.
    """
    
    return -np.dot(x,np.dot(inv_cov,x))/2.0


def init_cov(ndim):
    """
    Initialise random non-diagonal covariance matrix.
    Args: 
        - ndim: 
            Dimension of Gaussian.        
    Returns:
        - cov: 
            Covariance matrix of shape (ndim,ndim).
    """
    
    cov = np.zeros((ndim,ndim))
    diag_cov = np.ones(ndim) + np.random.randn(ndim)*0.1
    np.fill_diagonal(cov, diag_cov)
    
    for i in range(ndim-1):
        cov[i,i+1] = (-1)**i * 0.5*np.sqrt(cov[i,i]*cov[i+1,i+1])
        cov[i+1,i] = cov[i,i+1]
    
    return cov


def run_example(ndim=2, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False):
    """
    Run Gaussian example with non-diagonal covariance matrix.
    Args: 
        - ndim: 
            Dimension of Gaussian.
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
            If True then display intermediate results.
        
    Returns:
        - None.
    """
    
    hm.logs.high_log('Non-diagonal Covariance Guassian example')
    hm.logs.high_log('Dimensionality = {}'.format(ndim))
    hm.logs.low_log('---------------------------------')
    savefigs = True

    # Initialise covariance matrix.
    cov = init_cov(ndim)
    inv_cov = np.linalg.inv(cov)    
    hm.logs.low_log('Covariance matrix = {}'.format(cov))
    hm.logs.low_log('---------------------------------')
    # Start timer.
    clock = time.clock()
    
    # Run multiple realisations.
    n_realisations = 100
    evidence_inv_summary = np.zeros((n_realisations,3))
    for i_realisation in range(n_realisations):
        
        if n_realisations > 0:
            hm.logs.high_log('Realisation = {}/{}'
                .format(i_realisation, n_realisations))

        # Set up and run sampler.
        hm.logs.high_log('Run sampling...')
        hm.logs.low_log('---------------------------------')
        pos = np.random.rand(ndim * nchains).reshape((nchains, ndim))
        hm.logs.low_log('pos.shape = {}'.format(pos.shape))
        sampler = emcee.EnsembleSampler(nchains, ndim, \
                                        ln_posterior, args=[inv_cov])
        rstate = np.random.get_state() # Set random state to repeatable 
                                       # across calls.
        (pos, prob, state) = sampler.run_mcmc(pos, samples_per_chain,  
                                              rstate0=rstate) 
        samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
        lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])
            
        # Calculate evidence using harmonic....
        
        # Set up chains.
        chains = hm.Chains(ndim)
        chains.add_chains_3d(samples, lnprob)
        chains_train, chains_test = hm.utils.split_data(chains, \
                                                    training_proportion=0.05)
        hm.logs.low_log('---------------------------------')
        
        # Fit model.
        hm.logs.high_log('Fit model...')
        hm.logs.low_log('---------------------------------')
        r_scale = np.sqrt(ndim-1)
        hm.logs.low_log('r scale = {}'.format(r_scale))
        domains = [r_scale*np.array([1E-1,1E0])]
        hm.logs.low_log('Domain = {}'.format(domains))
        model = hm.model.HyperSphere(ndim, domains)
        fit_success, objective = model.fit(chains_train.samples, \
                                           chains_train.ln_posterior)        
        hm.logs.low_log('model.R = {}'.format(model.R))    
        # model.set_R(1.0)
        # if verbose: print("model.R = {}\n".format(model.R))
        hm.logs.low_log('Fit success = {}'.format(fit_success))    
        hm.logs.low_log('Objective = {}'.format(objective))    
        hm.logs.low_log('---------------------------------')
        # Use chains and model to compute inverse evidence.
        hm.logs.high_log('Compute evidence...')
        ev = hm.Evidence(chains_test.nchains, model)    
        # ev.set_mean_shift(0.0)
        ev.add_chains(chains_test)
        ln_evidence, ln_evidence_std = ev.compute_ln_evidence()

        # Compute analytic evidence.
        if i_realisation == 0:
            ln_evidence_analytic = ln_analytic_evidence(ndim, cov)

        # ======================================================================
        # Display evidence computation results.
        # ======================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('Evidence: analytic = {}, estimated = {}'
            .format(np.exp(ln_evidence_analytic), np.exp(ln_evidence)))
        hm.logs.low_log('Evidence: std = {}, std / estimate = {}'
            .format(np.exp(ln_evidence_std), \
                    np.exp(ln_evidence_std - ln_evidence)))
        diff = np.log(np.abs(np.exp(ln_evidence_analytic) - np.exp(ln_evidence)))
        hm.logs.high_log("Evidence: |analytic - estimate| / estimate = {}"
            .format(np.exp(diff - ln_evidence)))
        # ======================================================================
        # Display inverse evidence computation results.
        # ======================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('Inv Evidence: analytic = {}, estimate = {}'
            .format(np.exp(-ln_evidence_analytic), ev.evidence_inv))
        hm.logs.low_log('Inv Evidence: std = {}, std / estimate = {}'
            .format(np.sqrt(ev.evidence_inv_var), \
                    np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
        hm.logs.low_log('Inv Evidence: kurtosis = {}, \
                         sqrt( 2 / ( n_eff - 1 ) ) = {}'
            .format(ev.kurtosis, np.sqrt(2.0/(ev.n_eff-1))))     
        hm.logs.low_log('Inv Evidence: sqrt( var(var) ) / var = {}'
            .format(np.sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var))        
        hm.logs.high_log('Inv Evidence: |analytic - estimate| / estimate = {}'
            .format(np.abs(np.exp(-ln_evidence_analytic) \
                                  - ev.evidence_inv)/ev.evidence_inv))   
        # ======================================================================
        # Display more technical details for ln evidence.
        # ======================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('lnargmax = {}, lnargmin = {}'
            .format(ev.lnargmax, ev.lnargmin))
        hm.logs.low_log('lnprobmax = {}, lnprobmin = {}'
            .format(ev.lnprobmax, ev.lnprobmin))
        hm.logs.low_log('lnpredictmax = {}, lnpredictmin = {}'
            .format(ev.lnpredictmax, ev.lnpredictmin))
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('mean shift = {}, max shift = {}'
            .format(ev.mean_shift, ev.max_shift))
        hm.logs.low_log('running sum total = {}'
            .format(sum(ev.running_sum))) 
        hm.logs.low_log('running_sum = \n{}'
            .format(ev.running_sum))
        hm.logs.low_log('nsamples_per_chain = \n{}'
            .format(ev.nsamples_per_chain))
        hm.logs.low_log('nsamples_eff_per_chain = \n{}'
            .format(ev.nsamples_eff_per_chain))
        hm.logs.low_log('===============================')
        # ======================================================================

        # ======================================================================
        # Create corner/triangle plot.
        # ======================================================================
        if plot_corner and i_realisation == 0:
            
            utils.plot_corner(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('examples/plots/gaussian_nondiagcov_corner.png',
                            bbox_inches='tight')
            
            utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('examples/plots/gaussian_nondiagcov_getdist.png',
                            bbox_inches='tight')
                    
            plt.show()        
            
        # ======================================================================
        # In 2D case, plot surface/image and samples.  
        # ======================================================================
        if plot_surface and ndim == 2 and i_realisation == 0:
                    
            # ==================================================================
            # Define plot parameters.  
            # ==================================================================
            nx = 50
            xmin = -3.0
            xmax = 3.0

            # ==================================================================
            # 2D surface plot of posterior. 
            # ==================================================================
            ln_posterior_func = partial(ln_posterior, inv_cov=inv_cov)
            ln_posterior_grid, x_grid, y_grid = \
                utils.eval_func_on_grid(ln_posterior_func, 
                                        xmin=xmin, xmax=xmax, 
                                        ymin=xmin, ymax=xmax, 
                                        nx=nx, ny=nx)
            i_chain = 0
            ax = utils.plot_surface(np.exp(ln_posterior_grid), x_grid, y_grid, 
                                    samples[i_chain,:,:].reshape((-1, ndim)), 
                                    np.exp(lnprob[i_chain,:].reshape((-1, 1))),
                                    contour_z_offset=-0.5)
            # ax.set_zlim(-100.0, 0.0)                
            ax.set_zlabel(r'$\mathcal{L}$') 

            # Save.
            if savefigs:
                plt.savefig('examples/plots/gaussian_nondiagcov_posterior_surface.png'\
                    , bbox_inches='tight')

            plt.show(block=False)

            # ==================================================================
            # Image of posterior samples overlayed with contour plot.
            # ================================================================== 
            # Plot posterior image.
            ax = utils.plot_image(np.exp(ln_posterior_grid), x_grid, y_grid, 
                                  samples[i_chain].reshape((-1, ndim)),
                                  colorbar_label='$\mathcal{L}$',
                                  plot_contour=True)
            # Save.
            if savefigs:
                plt.savefig('examples/plots/gaussian_nondiagcov_posterior_image.png' \
                    , bbox_inches='tight')

            plt.show(block=False) 
        
            # ==================================================================
            # Learnt model of the posterior 
            # ================================================================== 
            # Evaluate ln_posterior and model over grid.
            x = np.linspace(xmin, xmax, nx); y = np.linspace(xmin, xmax, nx)
            x, y = np.meshgrid(x, y)     
            ln_model_grid = np.zeros((nx,nx))      
            for i in range(nx):
                for j in range(nx):
                    ln_model_grid[i,j]=model.predict(np.array([x[i,j],y[i,j]]))

            i_chain = 0
            ax = utils.plot_surface(np.exp(ln_model_grid), x_grid, y_grid, 
                                    #samples[i_chain,:,:].reshape((-1, ndim)), 
                                    #np.exp(lnprob[i_chain,:].reshape((-1, 1))),
                                    contour_z_offset=-0.075)
            # ax.set_zlim(-100.0, 0.0)                
            ax.set_zlabel(r'$\mathcal{L}$') 

            # Save.
            if savefigs:
                plt.savefig('examples/plots/gaussian_nondiagcov_surface.png' \
                    , bbox_inches='tight')

            plt.show(block=False)

            # ==================================================================
            # Projection of posteior onto x1,x2 plane with contours.
            # ================================================================== 
            # Plot posterior image.
            ax = utils.plot_image(np.exp(ln_model_grid), x_grid, y_grid, 
                                  # samples[i_chain].reshape((-1, ndim)),
                                  colorbar_label='$\mathcal{L}$',
                                  plot_contour=True)
            # Save.
            if savefigs:
                plt.savefig('examples/plots/gaussian_nondiagcov_image.png', 
                            bbox_inches='tight')

            plt.show(block=False) 
            # ================================================================== 
        
        evidence_inv_summary[i_realisation,0] = ev.evidence_inv
        evidence_inv_summary[i_realisation,1] = ev.evidence_inv_var
        evidence_inv_summary[i_realisation,2] = ev.evidence_inv_var_var
        
    clock = time.clock() - clock
    hm.logs.high_log('Execution_time = {}s'.format(clock))

    if n_realisations > 1:
        np.savetxt("examples/data/gaussian_nondiagcov_evidence_inv" +
                   "_realisations.dat",
                   evidence_inv_summary)
        evidence_inv_analytic_summary = np.zeros(1)
        evidence_inv_analytic_summary[0] = np.exp(-ln_evidence_analytic)
        np.savetxt("examples/data/gaussian_nondiagcov_evidence_inv" +
                   "_analytic.dat",
                   evidence_inv_analytic_summary)
    
    created_plots = True 
    if created_plots:
        input("\nPress Enter to continue...")
        

if __name__ == '__main__':
    
    # Define parameters.
    ndim = 2
    nchains = 100
    samples_per_chain = 5000
    nburn = 500     
    np.random.seed(10)
    
    # Run example.
    run_example(ndim, nchains, samples_per_chain, nburn, 
                plot_corner=True, plot_surface=True, verbose=False)