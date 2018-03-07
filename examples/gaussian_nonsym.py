import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
import emcee
#import scipy.special as sp
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
    """Run Gaussian example with diagonal covariance matrix.

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

    # Initialise covariance matrix.
    cov = init_cov(ndim)
    inv_cov = np.linalg.inv(cov)    
    if verbose: print("Covariance matrix = \n{}".format(cov))

    # Start timer.
    clock = time.clock()

    # Set up and run sampler.        
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
    r_scale = np.sqrt(ndim-1)
    if verbose: print("r_scale = {}".format(r_scale))
    domains = [r_scale*np.array([1E-1,1E1])]
    if verbose: print("domains = {}".format(domains))
    model = hm.model.HyperSphere(ndim, domains)
    model.fit(chains_train.samples, chains_train.ln_posterior)    
    if verbose: print("model.R = {}".format(model.R))    
    
    # Using chains and model to compute inverse evidence.
    ev = hm.Evidence(chains_test.nchains, model)
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
          
    if verbose: print("evidence_inv_analytic = {}"
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

    # Create corner/triangle plot.
    if plot_corner:
        
        names = ["x%s"%i for i in range(ndim)]
        labels =  ["x_%s"%i for i in range(ndim)]
        labels_corner =  ["$x_%s$"%i for i in range(ndim)]
        
        # Plot using corner.
        import corner        
        fig = corner.corner(samples.reshape((-1, ndim)), labels=labels_corner)
        #plt.savefig('corner.png', bbox_inches='tight')        
        
        # Plot using getdist.
        from getdist import plots, MCSamples
        import getdist        
        mcsamples = MCSamples(samples=samples.reshape((-1, ndim)), 
                              names=names, labels=labels)        
        g = plots.getSubplotPlotter()
        g.triangle_plot([mcsamples], filled=True)
        #plt.savefig('getdist.png', bbox_inches='tight')
        
        plt.show()        
        
    # In 2D case, plot surface and samples.    
    if plot_surface and ndim == 2:
        
        from mpl_toolkits.mplot3d import Axes3D
        
        # Define plot parameters.
        nx = 50
        xmin = -3.0
        xmax = 3.0
        
        # Evaluate ln_posterior over grid.
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(xmin, xmax, nx)
        x, y = np.meshgrid(x, y)
        z = np.zeros((nx,nx))        
        for i in range(nx):
            for j in range(nx):
                z[i,j] = ln_posterior(np.array([x[i,j],y[i,j]]), inv_cov)
        
        # Set up axis.
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        
        # Plot surface.
        # ls = LightSource(270, 45)
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        #rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')


        # Get lighting object for shading surface plots.
        from matplotlib.colors import LightSource


        # Create an instance of a LightSource and use it to illuminate the surface.
        light = LightSource(60, 120)
        illuminated_surface = light.shade(np.exp(z), cmap=cm.coolwarm)

        
        rgb = np.ones((z.shape[0], z.shape[1], 3))
        illuminated_surface = light.shade_rgb(rgb * np.array([0,0.0,1.0]), np.exp(z))



        ax.plot_surface(x, y, np.exp(z), alpha=0.3, linewidth=0, antialiased=False, facecolors=illuminated_surface)
        
        
        
        
        
        
        cset = ax.contour(x, y, np.exp(z), zdir='z', offset=-0.5, cmap=cm.coolwarm)
        # cset = ax.contourf(x, y, np.exp(z), zdir='z', offset=-0.5, cmap=cm.coolwarm)
        
        # cset = ax.contour(x, y, np.exp(z), zdir='x', offset=-4, cmap=cm.coolwarm)
        # cset = ax.contour(x, y, np.exp(z), zdir='y', offset=4, cmap=cm.coolwarm)
        # 
        
        
        # Plot samples (for chain 0 only).
        i_chain = 0
        xplot = samples[i_chain,:,1].reshape((-1, ndim))
        yplot = samples[i_chain,:,0].reshape((-1, ndim))        
        # Manually remove samples outside of plot region 
        # (since Matplotlib clipping cannot do this in 3D; see 
        # https://github.com/matplotlib/matplotlib/issues/749).
        xplot[xplot < xmin] = np.nan
        xplot[xplot > xmax] = np.nan        
        yplot[yplot < xmin] = np.nan
        yplot[yplot > xmax] = np.nan
        
        zplot = np.exp(lnprob[i_chain,:].reshape((-1, 1))) 
        
         
        ax.scatter(xplot,
                   yplot,
                   zplot, 
                   c='r', s=5, marker='.')
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.set_zlim(-0.5, 1.0)
        
        # ax.scatter(samples[0,:,0], samples[0,:,1], np.exp(lnprob[0,:]), c='r', s=5, marker='.')
        
        # ax.set_zlim(-4.0,0.0)
        
        ax.view_init(elev=15.0, azim=110.0)
        
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        
        plt.savefig('surface.png', bbox_inches='tight')
        
        plt.show()
        
                    
        # Temporary code to add more chains    
        # sampler.reset()
        # (pos, prob, state) = sampler.run_mcmc(pos, 10*samples_per_chain)
        # samples = np.ascontiguousarray(sampler.chain[:,:,:])
        # lnprob = np.ascontiguousarray(sampler.lnprobability[:,:])
        # 
        # 
        # chains2 = hm.Chains(ndim)
        # chains2.add_chains_3d(samples, lnprob)
        # 
        # chains_train2, chains_test2 = hm.utils.split_data(chains2, training_proportion=0.01)
        # 
        # ev.add_chains(chains_test2)
        # 
        # 
        # print("ln_rho_est = ", np.log(ev.evidence_inv), \
        #     " rel error = ", np.sqrt(ev.evidence_inv_var)/ev.evidence_inv, "(in linear space)")
                

    clock = time.clock() - clock


if __name__ == '__main__':
    
    # Define parameters.
    ndim = 2
    nchains = 100
    samples_per_chain = 5000
    nburn = 500     
    np.random.seed(10)
    
    # Run example.
    run_example(ndim, nchains, samples_per_chain, nburn, 
                plot_corner=True, plot_surface=True, verbose=True)
    