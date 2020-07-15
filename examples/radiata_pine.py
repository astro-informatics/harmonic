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


def ln_likelihood(y, x, n, alpha, beta, tau):
    """Compute log_e of Radiata Pine likelihood.

    Args:

        y: Compression strength along grain.

        x: Predictor (density or density adjusted for resin content).

        alpha: Model bias term.

        beta: Model linear term.

        tau: Prior precision factor.

    Returns:

        double: Value of log_e likelihood at specified point in parameter
            space.

    """

    ln_like = 0.5 * n * np.log(tau)
    ln_like -= 0.5 * n * np.log(2.0 * np.pi)
    
    s = np.sum((y - alpha - beta * x)**2)
    
    ln_like -= 0.5 * tau * s
    
    return ln_like

    
def ln_prior_alpha(alpha, tau, mu_0, r_0):
    """Compute log_e of alpha / beta prior (Normal prior).

    Args:

        alpha: Model term (bias or linear term).

        tau: Prior precision factor.

        mu_0: Prior mean.

        r_0: Prior precision constant factor.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """
    ln_pr_alpha = 0.5 * np.log(tau)
    ln_pr_alpha += 0.5 * np.log(r_0)
    ln_pr_alpha -= 0.5 * np.log(2.0 * np.pi)
    ln_pr_alpha -= 0.5 * tau * r_0 * (alpha - mu_0)**2
    
    return ln_pr_alpha


def ln_prior_tau(tau, a_0, b_0):
    """Compute log_e of tau prior (Gamma prior).

    Args:

        tau: Prior precision factor.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e tau prior at specified point in parameter
            space.

    """

    if tau < 0:
        return -np.inf
    
    ln_pr_tau = a_0 * np.log(b_0)
    ln_pr_tau += (a_0 - 1.0) * np.log(tau)
    ln_pr_tau -= b_0 * tau
    ln_pr_tau -= sp.gammaln(a_0)

    return ln_pr_tau


def ln_prior_separated(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of prior (combining individual prior functions).

    Args:

        alpha: Model bias term.

        beta: Model linear term.

        tau: Prior precision factor.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """
    ln_pr = ln_prior_alpha(alpha, tau, mu_0[0,0], r_0)
    ln_pr += ln_prior_alpha(beta, tau, mu_0[1,0], s_0)
    ln_pr += ln_prior_tau(tau, a_0, b_0)
        
    return ln_pr


def ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of combined prior (jointly computing total prior).

    Args:

        alpha: Model bias term.

        beta: Model linear term.

        tau: Prior precision factor.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """
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
    """Compute log_e of combined prior.

    Can be used to easily switch with prior function using (e.g.
    ln_prior_separated or ln_prior_combined). There should be (and is) not
    difference (both implemented just as an additional consistency check).

    Args:

        alpha: Model bias term.

        beta: Model linear term.

        tau: Prior precision factor.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """

    return ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)
        

def ln_posterior(theta, y, x, n, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of posterior.
    
    Args:

        theta: Position (alpha, beta, tau) at which to evaluate posterior.

        y: Compression strength along grain.

        x: Predictor (density or density adjusted for resin content).

        n: Number of specimens.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e posterior at specified theta (alpha, beta,
            tau) point.

    """
    
    alpha, beta, tau = theta

    ln_pr = ln_prior(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)
    
    if not np.isfinite(ln_pr):
        return -np.inf

    ln_L = ln_likelihood(y, x, n, alpha, beta, tau)    
    
    return  ln_L + ln_pr
    

def ln_evidence_analytic(x, y, n, mu_0, r_0, s_0, a_0, b_0):
    """Compute log_e of analytic evidence.

    Args:

        x: Predictor (density or density adjusted for resin content).

        y: Compression strength along grain.

        n: Number of specimens.

        mu_0: Prior means.

        r_0: Prior precision constant factor for bias term.

        s_0: Prior precision constant factor for linear term.

        a_0: Gamma prior shape parameter.

        b_0: Gamma prior rate parameter.

    Returns:

        double: Value of log_e of analytic evidence for model.

    """

    Q_0 = np.diag([r_0, s_0])
    X = np.c_[np.ones((n, 1)), x]
    M = X.T.dot(X) + Q_0
    nu_0 = np.linalg.inv(M).dot(X.T.dot(y) + Q_0.dot(mu_0))

    quad_terms = y.T.dot(y) + mu_0.T.dot(Q_0).dot(mu_0) - \
                        nu_0.T.dot(M).dot(nu_0)

    ln_evidence = -0.5 * n * np.log(np.pi)
    ln_evidence += a_0 * np.log(2.0*b_0)
    ln_evidence += sp.gammaln(0.5*n + a_0) - sp.gammaln(a_0)
    ln_evidence += 0.5 * np.log(np.linalg.det(Q_0)) - \
               0.5 * np.log(np.linalg.det(M))

    ln_evidence += -(0.5 * n + a_0) * np.log(quad_terms + 2.0 * b_0)

    return ln_evidence

    
def run_example(model_1=True, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False):
    """Run Radiata Pine example.

    Args:

        model_1: Consider model 1 if true, otherwise model 2.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        nburn: Number of burn in samples for each chain.

        verbose: If True then display intermediate results.

        plot_corner: Plot marginalised distributions if true.

        plot_surface: Plot surface and samples if true.

    """
       
    hm.logs.info_log('Radiata Pine example')
    ndim=3
    hm.logs.info_log('Dimensionality = {}'.format(ndim))
         
    # Set general parameters.    
    savefigs = True
    
    nfold = 3
    training_proportion = 0.25
    hyper_parameters_sphere = [None]
    domains_sphere = [np.array([1E-1,5E0])]
    
    #===========================================================================
    # Set-up Priors
    #===========================================================================
    # Define prior variables
    mu_0 = np.array([[3000.0], [185.0]])    
    r_0 = 0.06
    s_0 = 6.0
    a_0 = 3.0
    b_0 = 2.0 * 300**2
    
    #===========================================================================
    # Load Radiata Pine data.
    #===========================================================================
    hm.logs.info_log('Loading data ...')

    # Imports data file
    data = np.loadtxt('examples/data/RadiataPine.dat')
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

    #===========================================================================
    # Compute random positions to draw from for emcee sampler.
    #===========================================================================
    """
    Initial positions for each chain for each covariate \in [0,8).
    Simply drawn from directly from each covariate prior.
    """
    pos_alpha = mu_0[0,0] + \
                1.0 / np.sqrt(tau_prior_mean * r_0) * np.random.randn(nchains)  
    pos_beta = mu_0[1,0] + \
               1.0 / np.sqrt(tau_prior_mean * s_0) * np.random.randn(nchains)              
    pos_tau = tau_prior_mean + tau_prior_std * \
                           (np.random.rand(nchains) - 0.5)  # avoid negative tau

    """
    Concatenate these positions into a single variable 'pos'.
    """  
    pos = np.c_[pos_alpha, pos_beta, pos_tau]
     
    # Start timer.
    clock = time.clock()

    #===========================================================================
    # Run Emcee to recover posterior sampels 
    #===========================================================================
    hm.logs.info_log('Run sampling...')
    """
    Feed emcee the ln_posterior function, starting positions and recover chains.
    """
    if model_1:
        args = (y, x, n, mu_0, r_0, s_0, a_0, b_0)
    else:
        args = (y, z, n, mu_0, r_0, s_0, a_0, b_0)
    sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, \
        args=args)
    rstate = np.random.get_state()
    sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
    samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
    lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

    #===========================================================================
    # Configure emcee chains for harmonic
    #===========================================================================
    hm.logs.info_log('Configure chains...')
    """
    Configure chains for the cross validation stage.
    """
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, lnprob)
    chains_train, chains_test = hm.utils.split_data(chains, \
        training_proportion=training_proportion)
        
    #===========================================================================
    # Fit learnt model for container function 
    #===========================================================================
    hm.logs.info_log('Select model...')
    """
    This could simply use the cross-validation results to choose the model which 
    has the smallest validation variance -- i.e. the best model for the job. Here
    however we manually select the hypersphere model.
    """

    hm.logs.info_log('Using HyperSphere')
    model = hm.model.HyperSphere(ndim, domains_sphere, hyper_parameters=None)            
        
    fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
    hm.logs.debug_log('fit_success = {}'.format(fit_success))    
    
    model.set_R(model.R) # conservative reduction in R.
    hm.logs.debug_log('model.R = {}'.format(model.R))
    
    #===========================================================================
    # Computing evidence using learnt model and emcee chains
    #===========================================================================
    hm.logs.info_log('Compute evidence...')
    """
    Instantiates the evidence class with a given model. Adds some chains and 
    computes the log-space evidence (marginal likelihood).
    """
    ev = hm.Evidence(chains_test.nchains, model)
    ev.add_chains(chains_test)
    ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
    evidence_std_log_space = np.log(np.exp(ln_evidence) + np.exp(ln_evidence_std)) - ln_evidence

    #===========================================================================
    # End Timer.
    clock = time.clock() - clock
    hm.logs.info_log('execution_time = {}s'.format(clock))
    
    #===========================================================================
    # Display evidence results 
    #===========================================================================
    hm.logs.info_log('ln_evidence = {} +/- {}'.format(ln_evidence, evidence_std_log_space))
    hm.logs.info_log('kurtosis = {}'.format(ev.kurtosis))
    hm.logs.info_log('sqrt( 2/(n_eff-1) ) = {}'.format(np.sqrt(2.0/(ev.n_eff-1))))
    check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
    hm.logs.info_log('sqrt(evidence_inv_var_var) / evidence_inv_var = {}'.format(check))
    ln_evidence_analytic_model1 = \
        ln_evidence_analytic(x, y, n, mu_0, r_0, s_0, a_0, b_0)
    hm.logs.info_log('ln_evidence_analytic_model1 = {}'
                         .format(ln_evidence_analytic_model1[0][0]))
    ln_evidence_analytic_model2 = \
        ln_evidence_analytic(z, y, n, mu_0, r_0, s_0, a_0, b_0)
    hm.logs.info_log('ln_evidence_analytic_model2 = {}'
                         .format(ln_evidence_analytic_model2[0][0]))

    #===========================================================================
    # Display more technical details
    #===========================================================================
    hm.logs.debug_log('---------------------------------')
    hm.logs.critical_log('Technical Details')
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
    hm.logs.debug_log('statistic space = {}'.format(ev.statspace))
    hm.logs.debug_log('running sum total = {}'
        .format(sum(ev.running_sum)))
    hm.logs.debug_log('running sum = \n{}'
        .format(ev.running_sum))
    hm.logs.debug_log('nsamples per chain = \n{}'
        .format(ev.nsamples_per_chain))
    hm.logs.debug_log('nsamples eff per chain = \n{}'
        .format(ev.nsamples_eff_per_chain))
    hm.logs.debug_log('===============================')
   
    #===========================================================================
    # Plotting and prediction functions
    #===========================================================================

    # Create corner/triangle plot.
    created_plots = False
    if plot_corner:
        
        utils.plot_corner(samples.reshape((-1, ndim)))
        if savefigs:
            plt.savefig('examples/plots/radiatapine_corner.png',
                        bbox_inches='tight')
        
        utils.plot_getdist(samples.reshape((-1, ndim)))
        if savefigs:
            plt.savefig('examples/plots/radiatapine_getdist.png',
                        bbox_inches='tight')
        
        plt.show(block=False)  
        created_plots = True

    # Plot model over first two parameters.

    def model_predict_x0x1(x_2d):         
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
        plt.savefig('examples/plots/radiatapine_model_x0x1_image.png',
                    bbox_inches='tight')

    # Plot exponential of model.
    ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                          colorbar_label=r'$\varphi$')    
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')   
    #plt.axis('equal')    
    if savefigs:
        plt.savefig('examples/plots/radiatapine_modelexp_x0x1_image.png',
                    bbox_inches='tight')

    # Plot model over second and third parameters.

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
        plt.savefig('examples/plots/radiatapine_model_x1x2_image.png',
                    bbox_inches='tight')

    # Plot exponential of model.
    ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                          colorbar_label=r'$\varphi$')    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')   
    #plt.axis('equal')    
    if savefigs:
        plt.savefig('examples/plots/radiatapine_modelexp_x1x2_image.png',
                    bbox_inches='tight')

    # Plot model over first and third parameters.

    def model_predict_x0x2(x_2d): 
        x1 = 185.0
        x = np.append(x_2d[0], [x1])
        x = np.append(x, x_2d[1])
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
        plt.savefig('examples/plots/radiatapine_model_x0x2_image.png',
                    bbox_inches='tight')

    # Plot exponential of model.
    ax = utils.plot_image(np.exp(model_grid), x_grid, y_grid,
                          colorbar_label=r'$\varphi$')    
    plt.xlabel('$x_0$')
    plt.ylabel('$x_2$')   
    #plt.axis('equal')    
    if savefigs:
        plt.savefig('examples/plots/radiatapine_modelexp_x0x2_image.png',
                    bbox_inches='tight')

    plt.show(block=False)  

    if created_plots:
        input("\nPress Enter to continue...")


if __name__ == '__main__':

    # Setup logging config.
    hm.logs.setup_logging()

    # Define parameters.
    model_1 = False
    nchains = 400
    samples_per_chain = 20000
    nburn = 2000
    np.random.seed(2)
    
    # Run example.
    samples = run_example(model_1, nchains, samples_per_chain, nburn, 
                          plot_corner=True, plot_surface=True,
                          verbose=True)

