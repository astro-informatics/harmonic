import numpy as np
import sys
import emcee
import time 
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.append(".")
import harmonic as hm
sys.path.append("examples")
import utils

# Setup Logging config
hm.logs.setup_logging()


def ln_analytic_evidence(ndim, cov):
    """Compute analytic evidence for nD Gaussian.
    
    Args:
        ndim: Dimension of Gaussian.
        cov: Covariance matrix.
    
    Returns:
        double: Analytic evidence.
    """
    
    ln_norm_lik = 0.5*ndim*np.log(2*np.pi) + 0.5*np.log(np.linalg.det(cov))
    return ln_norm_lik


def ln_posterior(x, inv_cov):
    """Compute log_e of posterior.
    
    Args: 
        x: Position at which to evaluate posterior.
        inv_cov: Inverse covariance matrix.    
        
    Returns:
        double: Value of Gaussian at specified point.
    """
    
    return -np.dot(x,np.dot(inv_cov,x))/2.0


def init_cov(ndim):
    """Initialise random non-diagonal covariance matrix covariance matrix.
    Args: 
        ndim: Dimension of Gaussian.        
        
    Returns:
        cov: Covariance matrix of shape (ndim,ndim).
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
    """Run Gaussian example with non-diagonal covariance matrix.
    Args: 
        ndim: Dimension of Gaussian.
        nchains: Number of chains.
        samples_per_chain: Number of samples per chain.
        nburn: Number of burn in samples.
        plot_corner: Plot marginalised distributions if true.
        plot_surface: Plot surface and samples if true.
        verbose: If True then display intermediate results.
        
    Returns:
        None.
    """
    
    hm.logs.high_log('nD Guassian example')
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
        chains_train, chains_test = hm.utils.split_data(chains, 
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
            .format(np.exp(ln_evidence_std), 
                np.exp(ln_evidence_std - ln_evidence)))
        diff = np.log(np.abs(np.exp(ln_evidence_analytic) - \
            np.exp(ln_evidence)))
        hm.logs.high_log("Evidence: |analytic - estimate| / estimate = {}"
            .format(np.exp(diff - ln_evidence)))
        # ======================================================================
        # Display inverse evidence computation results.
        # ======================================================================
        hm.logs.low_log('---------------------------------')
        hm.logs.low_log('Inv Evidence: analytic = {}, estimate = {}'
            .format(np.exp(-ln_evidence_analytic), ev.evidence_inv))
        hm.logs.low_log('Inv Evidence: std = {}, std / estimate = {}'
            .format(np.sqrt(ev.evidence_inv_var), 
                np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
        hm.logs.low_log('Inv Evidence: kurtosis = {},\
                      sqrt( 2 / ( n_eff - 1 ) ) = {}'
            .format(ev.kurtosis, np.sqrt(2.0/(ev.n_eff-1))))     
        hm.logs.low_log('Inv Evidence: sqrt( var(var) ) / var = {}'
            .format(np.sqrt(ev.evidence_inv_var_var)/ev.evidence_inv_var))        
        hm.logs.high_log('Inv Evidence: |analytic - estimate| / estimate = {}'
            .format(np.abs(np.exp(-ln_evidence_analytic) - \
                ev.evidence_inv)/ev.evidence_inv))
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
        hm.logs.low_log('running_sum_total = {}, mean_shift = {}'
            .format(sum(ev.running_sum), ev.mean_shift))   
        hm.logs.low_log('running_sum = \n{}'
            .format(ev.running_sum))
        hm.logs.low_log('nsamples_per_chain = \n{}'
            .format(ev.nsamples_per_chain))
        hm.logs.low_log('nsamples_eff_per_chain = \n{}'
            .format(ev.nsamples_eff_per_chain))
        hm.logs.low_log('===============================')
        # ======================================================================

        # Create corner/triangle plot.
        if plot_corner and i_realisation == 0:
            
            utils.plot_corner(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('./plots/gaussian_nondiagcov_corner.png',
                            bbox_inches='tight')
            
            utils.plot_getdist(samples.reshape((-1, ndim)))
            if savefigs:
                plt.savefig('./plots/gaussian_nondiagcov_getdist.png',
                            bbox_inches='tight')
                    
            plt.show()        
            
        # In 2D case, plot surface/image and samples.    
        if plot_surface and ndim == 2 and i_realisation == 0:
                    
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.colors import LightSource
            
            # Define plot parameters.
            nx = 50
            xmin = -3.0
            xmax = 3.0
            
            # Evaluate ln_posterior and model over grid.
            x = np.linspace(xmin, xmax, nx)
            y = np.linspace(xmin, xmax, nx)
            x, y = np.meshgrid(x, y)
            ln_posterior_grid = np.zeros((nx,nx))        
            ln_model_grid = np.zeros((nx,nx))      
            for i in range(nx):
                for j in range(nx):
                    ln_posterior_grid[i,j] = \
                        ln_posterior(np.array([x[i,j],y[i,j]]), inv_cov)
                    ln_model_grid[i,j] = \
                        model.predict(np.array([x[i,j],y[i,j]]))
            
            # Set up axis for surface plot.
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))        

            # Create an instance of a LightSource and use it to illuminate
            # the surface.
            light = LightSource(60, 120)
            rgb = np.ones((ln_posterior_grid.shape[0], 
                           ln_posterior_grid.shape[1], 3))
            illuminated_surface = \
                light.shade_rgb(rgb * np.array([0,0.0,1.0]), 
                                np.exp(ln_posterior_grid))

            # Plot surface.
            ax.plot_surface(x, y, np.exp(ln_posterior_grid), 
                            alpha=0.3, linewidth=0, antialiased=False, 
                            facecolors=illuminated_surface)
            
            # Plot contour.
            cset = ax.contour(x, y, np.exp(ln_posterior_grid), 
                              zdir='z', offset=-0.5, cmap=cm.coolwarm)        
            
            # Plot samples (for chain 0 only).
            i_chain = 0
            xplot = samples[i_chain,:,0].reshape((-1, ndim))
            yplot = samples[i_chain,:,1].reshape((-1, ndim))        
            # Manually remove samples outside of plot region 
            # (since Matplotlib clipping cannot do this in 3D; see 
            # https://github.com/matplotlib/matplotlib/issues/749).
            xplot[xplot < xmin] = np.nan
            xplot[xplot > xmax] = np.nan        
            yplot[yplot < xmin] = np.nan
            yplot[yplot > xmax] = np.nan        
            zplot = np.exp(lnprob[i_chain,:].reshape((-1, 1)))                  
            ax.scatter(xplot, yplot, zplot, c='r', s=5, marker='.')
            
            # Define additional plot settings.
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)
            ax.set_zlim(-0.5, 1.0)
            ax.view_init(elev=15.0, azim=110.0)        
            ax.set_xlabel('$x_0$')
            ax.set_ylabel('$x_1$')
            
            # Save.
            if savefigs:
                plt.savefig('./plots/gaussian_nondiagcov_posterior_surface.png',
                 bbox_inches='tight')
                    
            # Create image plot of posterior.
            plt.figure()
            plt.imshow(np.exp(ln_posterior_grid), origin='lower', 
                       extent=[xmin, xmax, xmin, xmax])
            plt.contour(x, y, np.exp(ln_posterior_grid), cmap=cm.coolwarm)
            plt.plot(samples[i_chain,:,0].reshape((-1, ndim)), 
                     samples[i_chain,:,1].reshape((-1, ndim)), 
                     'r.', markersize=1)
            plt.colorbar()
            plt.xlabel('$x_0$')
            plt.ylabel('$x_1$')     
                       
            if savefigs:
                plt.savefig('./plots/gaussian_nondiagcov_posterior_image.png', 
                            bbox_inches='tight')        
            
            # Create surface plot of model.
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            
            illuminated_surface = \
                light.shade_rgb(rgb * np.array([0,0.0,1.0]), 
                                np.exp(ln_model_grid))                            
            
            ax.plot_surface(x, y, np.exp(ln_model_grid), 
                            alpha=0.3, linewidth=0, antialiased=False, 
                            facecolors=illuminated_surface)
            
            cset = ax.contour(x, y, np.exp(ln_model_grid), zdir='z', 
                              offset=-0.075, cmap=cm.coolwarm)
            
            ax.view_init(elev=15.0, azim=110.0)        
            ax.set_xlabel('$x_0$')
            ax.set_ylabel('$x_1$')
            
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)
            ax.set_zlim(-0.075, 0.30)
            
            if savefigs:
                plt.savefig('./plots/gaussian_nondiagcov_surface.png', 
                            bbox_inches='tight')
                    
            # Create image plot of model.
            plt.figure()        
            plt.imshow(np.exp(ln_model_grid), origin='lower', 
                       extent=[xmin, xmax, xmin, xmax])
            plt.contour(x, y, np.exp(ln_model_grid), cmap=cm.coolwarm)
            plt.colorbar()
            plt.xlabel('$x_0$')
            plt.ylabel('$x_1$')
            if savefigs:
                plt.savefig('./plots/gaussian_nondiagcov_image.png', 
                            bbox_inches='tight')
                        
            plt.show(block=False)
        
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
                plot_corner=False, plot_surface=False, verbose=False)