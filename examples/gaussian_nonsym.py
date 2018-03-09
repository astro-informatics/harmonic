import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
import emcee
import time 
import matplotlib.pyplot as plt
from matplotlib import cm

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
    
    print("nD Guassian example")
    print("ndim = {}".format(ndim))

    savefigs = False

    # Initialise covariance matrix.
    cov = init_cov(ndim)
    inv_cov = np.linalg.inv(cov)    
    if verbose: print("Covariance matrix = \n{}".format(cov))

    # Start timer.
    clock = time.clock()

    # Set up and run sampler.
    print("Run sampling...")
    pos = np.random.rand(ndim * nchains).reshape((nchains, ndim))
    if verbose: print("pos.shape = {}".format(pos.shape))
    sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=[inv_cov])
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
    
    # Fit model.
    print("Fit model...")
    r_scale = np.sqrt(ndim-1)
    if verbose: print("r_scale = {}".format(r_scale))
    domains = [r_scale*np.array([1E-1,1E0])]
    if verbose: print("domains = {}".format(domains))
    model = hm.model.HyperSphere(ndim, domains)
    fit_success, objective = model.fit(chains_train.samples, chains_train.ln_posterior)        
    if verbose: print("model.R = {}".format(model.R))    
    # model.set_R(1.0)
    # if verbose: print("model.R = {}\n".format(model.R))
    if verbose: print("fit_success = {}".format(fit_success))    
    if verbose: print("objective = {}\n".format(objective))    
        
    # Use chains and model to compute inverse evidence.
    print("Compute evidence...")
    ev = hm.Evidence(chains_test.nchains, model)    
    # ev.set_mean_shift(0.0)
    ev.add_chains(chains_test)
    ln_evidence, ln_evidence_std = ev.compute_ln_evidence()

    # Compute analytic evidence.
    ln_evidence_analytic = ln_analytic_evidence(ndim, cov)

    # Display results.
    print("evidence_analytic = {}".format(np.exp(ln_evidence_analytic)))
    print("evidence = {}".format(np.exp(ln_evidence)))
    print("evidence_std = {}".format(np.exp(ln_evidence_std)))
    print("evidence_std / evidence = {}"
          .format(np.exp(ln_evidence_std - ln_evidence)))
    diff = np.log(np.abs(np.exp(ln_evidence_analytic) - np.exp(ln_evidence)))
    print("|evidence_analytic - evidence| / evidence = {}"
          .format(np.exp(diff - ln_evidence)))
          
    if verbose: print("\nevidence_inv_analytic = {}"
        .format(np.exp(-ln_evidence_analytic)))
    if verbose: print("evidence_inv = {}"
        .format(ev.evidence_inv))
    if verbose: print("evidence_inv_std = {}"
        .format(np.sqrt(ev.evidence_inv_var)))
    if verbose: print("evidence_inv_std / evidence_inv = {}"
        .format(np.sqrt(ev.evidence_inv_var)/ev.evidence_inv))
    if verbose: 
        print("|evidence_analytic_inv - evidence_inv| / evidence_inv = {}"
            .format(np.abs(np.exp(-ln_evidence_analytic) - ev.evidence_inv)/ev.evidence_inv))

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
    if verbose: print("nsamples_eff_per_chain = \n{}"
        .format(ev.nsamples_eff_per_chain))

    # Create corner/triangle plot.
    if plot_corner:
        
        names = ["x%s"%i for i in range(ndim)]
        labels =  ["x_%s"%i for i in range(ndim)]
        labels_corner =  ["$x_%s$"%i for i in range(ndim)]
        
        # Plot using corner.
        import corner        
        fig = corner.corner(samples.reshape((-1, ndim)), 
                            labels=labels_corner)
        if savefigs:
            plt.savefig('./plots/corner.png', bbox_inches='tight')        
        
        # Plot using getdist.
        from getdist import plots, MCSamples
        import getdist        
        mcsamples = MCSamples(samples=samples.reshape((-1, ndim)), 
                              names=names, labels=labels)        
        g = plots.getSubplotPlotter()
        g.triangle_plot([mcsamples], filled=True)
        if savefigs:
            plt.savefig('./plots/getdist.png', bbox_inches='tight')
        
        plt.show()        
        
    # In 2D case, plot surface/image and samples.    
    if plot_surface and ndim == 2:
                
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
            plt.savefig('./plots/posterior_surface.png', bbox_inches='tight')
                
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
        # Save.
        if savefigs:
            plt.savefig('./plots/posterior_image.png', bbox_inches='tight')        
        
        # Create surface plot of model.
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        
        illuminated_surface = \
            light.shade_rgb(rgb * np.array([0,0.0,1.0]), 
                            np.exp(ln_model_grid))                            
        
        ax.plot_surface(x, y, np.exp(ln_model_grid), 
                        alpha=0.3, linewidth=0, antialiased=False, 
                        facecolors=illuminated_surface)
        
        cset = ax.contour(x, y, np.exp(ln_model_grid), zdir='z', offset=-0.075, 
                          cmap=cm.coolwarm)
        
        ax.view_init(elev=15.0, azim=110.0)        
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.set_zlim(-0.075, 0.30)
        
        if savefigs:
            plt.savefig('./plots/model_surface.png', bbox_inches='tight')
                
        # Create image plot of model.
        plt.figure()        
        plt.imshow(np.exp(ln_model_grid), origin='lower', 
                   extent=[xmin, xmax, xmin, xmax])
        plt.contour(x, y, np.exp(ln_model_grid), cmap=cm.coolwarm)
        plt.colorbar()
        plt.xlabel('$x_0$')
        plt.ylabel('$x_1$')
        if savefigs:
            plt.savefig('./plots/model_image.png', bbox_inches='tight')
                    
        plt.show()
        
    clock = time.clock() - clock
    print("execution_time = {}s".format(clock))

if __name__ == '__main__':
    
    # Define parameters.
    ndim = 5
    nchains = 100
    samples_per_chain = 5000
    nburn = 500     
    np.random.seed(10)
    
    # Run example.
    run_example(ndim, nchains, samples_per_chain, nburn, 
                plot_corner=False, plot_surface=False, verbose=False)
    