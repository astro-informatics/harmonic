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
    """
    Compute log_e of Radiatta Pine likelihood
    Args: 
        - y: 
            Vector of incidence. 1=diabetes, 0=no diabetes
        - x: 
            Vector of data covariates (e.g. NP, PGC, BP, TST, DMI e.t.c.).  
        - alpha:
            enter when known.
        - beta:
            enter when known.
        - tau:
            enter when known.
    Returns:
        - double: 
            Value of log_e likelihood at specified point in parameter space.
    """
    ln_like = 0.5 * n * np.log(tau)
    ln_like -= 0.5 * n * np.log(2.0 * np.pi)
    
    s = np.sum((y - alpha - beta * x)**2)
    
    ln_like -= 0.5 * tau * s
    
    return ln_like

    
def ln_prior_alpha(alpha, tau, mu_0, r_0):
    """
    Compute log_e of alpha / beta prior
    Args: 
        - alpha:
            enter when known.
        - tau:
            enter when known.
        - mu_0:
            enter when known.
        - r_0:
            enter when known.
    Returns:
        - double: 
            Value of log_e prior at specified point in parameter space.
    """
    ln_pr_alpha = 0.5 * np.log(tau)
    ln_pr_alpha += 0.5 * np.log(r_0)
    ln_pr_alpha -= 0.5 * np.log(2.0 * np.pi)
    ln_pr_alpha -= 0.5 * tau * r_0 * (alpha - mu_0)**2
    
    return ln_pr_alpha


def ln_prior_tau(tau, a_0, b_0):
    """
    Compute log_e of tau prior
    Args: 
        - tau:
            enter when known.
        - a_0:
            enter when known.
        - b_0:
            enter when known.
    Returns:
        - double: 
            Value of log_e tau prior at specified point in parameter space.
    """
    if tau < 0:
        return -np.inf
    
    ln_pr_tau = a_0 * np.log(b_0)
    ln_pr_tau += (a_0 - 1.0) * np.log(tau)
    ln_pr_tau -= b_0 * tau
    ln_pr_tau -= sp.gammaln(a_0)

    return ln_pr_tau


def ln_prior_separated(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    """
    Compute log_e of prior (combining individual prior functions)
    Args: 
        - alpha:
            enter when known.
        - tau:
            enter when known.
        - mu_0:
            enter when known.
        - r_0:
            enter when known.
         - s_0:
            enter when known.
        - a_0:
            enter when known.
        - b_0:
            enter when known.
    Returns:
        - double: 
            Value of log_e prior at specified point in parameter space.
    """
    ln_pr = ln_prior_alpha(alpha, tau, mu_0[0,0], r_0)
    ln_pr += ln_prior_alpha(beta, tau, mu_0[1,0], s_0)
    ln_pr += ln_prior_tau(tau, a_0, b_0)
        
    return ln_pr


def ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0):
    """
    Compute log_e of combined prior (jointly computing total prior)
    Args: 
        - alpha:
            enter when known.
        - tau:
            enter when known.
        - mu_0:
            enter when known.
        - r_0:
            enter when known.
         - s_0:
            enter when known.
        - a_0:
            enter when known.
        - b_0:
            enter when known.
    Returns:
        - double: 
            Value of log_e combined prior at specified point in parameter space.
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
    """
    Compute log_e of prior
    Args: 
        - alpha:
            enter when known.
        - tau:
            enter when known.
        - mu_0:
            enter when known.
        - r_0:
            enter when known.
         - s_0:
            enter when known.
        - a_0:
            enter when known.
        - b_0:
            enter when known.
    Returns:
        - double: 
            Value of log_e combined prior at specified point in parameter space.
    """

    return ln_prior_combined(alpha, beta, tau, mu_0, r_0, s_0, a_0, b_0)
        

def ln_posterior(theta, y, x, n, mu_0, r_0, s_0, a_0, b_0):
    """
    Compute log_e of posterior.
    
    Args: 
        - theta: 
            Position (alpha, beta, tau) at which to evaluate posterior.
        
    Returns:
        - double: 
            Value of log_e posterior at specified (alpha, beta, tau) point.
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
    
    
def ln_posterior_analytic_x(x, y, mu_0, a_0, b_0):
    """
    Compute analytic log_e of posterior for model x.
    
    Args: 
        
    Returns:
        - double: 
            Value of analytic log_e posterior for model x.
    """

    r_0 = 0.06
    s_0 = 6
    Q_0 = np.diag([r_0, s_0])
    n = len(x)

    X = np.c_[np.ones((n, 1)), x]
    M_x = X.T.dot(X) + Q_0

    beta0_x = np.linalg.inv(M_x).dot(X.T.dot(y) + Q_0.dot(mu_0))
    c0_x = y.T.dot(y) + mu_0.T.dot(Q_0).dot(mu_0) - \
                        beta0_x.T.dot(M_x).dot(beta0_x)

    ln_z2_x = -0.5 * n * np.log(np.pi)
    ln_z2_x += 0.5 * a_0 * np.log(b_0)
    ln_z2_x += sp.gammaln(0.5*(n + a_0)) - sp.gammaln(0.5*a_0)
    ln_z2_x += 0.5 * np.log(np.linalg.det(Q_0)) - \
               0.5 * np.log(np.linalg.det(M_x))

    ln_z2_x += -0.5 * (n + a_0) * np.log(c0_x + b_0)

    return ln_z2_x

def ln_posterior_analytic_z(z, y, mu_0, a_0, b_0):
    """
    Compute analytic log_e of posterior for model z.
    
    Args: 
        
    Returns:
        - double: 
            Value of analytic log_e posterior for model z.
    """

    r_0 = 0.06
    s_0 = 6
    Q_0 = np.diag([r_0, s_0])
    n = len(z)

    Z = np.c_[np.ones((n, 1)), z]
    M_z = Z.T.dot(Z) + Q_0

    beta0_z = np.linalg.inv(M_z).dot(Z.T.dot(y) + Q_0.dot(mu_0))
    c0_z = y.T.dot(y) + mu_0.T.dot(Q_0).dot(mu_0) - \
                        beta0_z.T.dot(M_z).dot(beta0_z)

    ln_z2_z = -0.5 * n * np.log(np.pi)
    ln_z2_z += 0.5 * a_0 * np.log(b_0)
    ln_z2_z += sp.gammaln(0.5*(n + a_0)) - sp.gammaln(0.5*a_0)
    ln_z2_z += 0.5 * np.log(np.linalg.det(Q_0)) - \
               0.5 * np.log(np.linalg.det(M_z))

    ln_z2_z += -0.5 * (n + a_0) * np.log(c0_z + b_0)

    return ln_z2_z

    
    
def run_example(model_1=True, nchains=100, samples_per_chain=1000, 
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False,
                plot_comparison=False):
    
    hm.logs.debug_log('---------------------------------')
    hm.logs.critical_log('Radiata Pine example')
    ndim=3
    hm.logs.critical_log('Dimensionality = {}'.format(ndim))
    hm.logs.debug_log('---------------------------------')
         
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
    hm.logs.critical_log('Loading data ...')
    hm.logs.debug_log('---------------------------------')

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

    # pos_alpha = mu_0[0,0]+ (np.random.rand(nchains)-0.5) * 0.1*mu_0[0,0]
    # pos_beta = mu_0[1,0]+ (np.random.rand(nchains)-0.5) * 0.1*mu_0[1,0]
    # pos_tau = tau_prior_mean + (np.random.rand(nchains)-0.5) * 0.1*tau_prior_mean
    
    """
    Concatenate these positions into a single variable 'pos'.
    """  
    pos = np.c_[pos_alpha, pos_beta, pos_tau]

    hm.logs.critical_log('Initial pos alpha: {}'.format(np.max(pos_alpha)))
    hm.logs.critical_log('Initial pos beta: {}'.format(np.max(pos_beta)))
    hm.logs.critical_log('Initial pos tau: {}'.format(np.max(pos_tau)))
           
    # Start timer.
    clock = time.clock()

    #===========================================================================
    # Run Emcee to recover posterior sampels 
    #===========================================================================
    hm.logs.critical_log('Run sampling...')
    hm.logs.debug_log('---------------------------------')
    """
    Feed emcee the ln_posterior function, starting positions and recover chains.
    """
    if model_1:
        args = (y, z, n, mu_0, r_0, s_0, a_0, b_0)
    else:
        args = (y, x, n, mu_0, r_0, s_0, a_0, b_0)
    sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, \
        args=args)
    rstate = np.random.get_state()
    sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
    samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
    lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

    #===========================================================================
    # Configure emcee chains for harmonic
    #===========================================================================
    hm.logs.critical_log('Configure chains...')
    hm.logs.debug_log('---------------------------------')
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
    hm.logs.critical_log('Select model...')
    hm.logs.debug_log('---------------------------------')
    """
    This could simply use the cross-validation results to choose the model which 
    has the smallest validation variance -- i.e. the best model for the job. Here
    however we manually select the hypersphere model.
    """

    hm.logs.critical_log('Using HyperSphere')
    model = hm.model.HyperSphere(ndim, domains_sphere, hyper_parameters=None)            
        
    fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
    hm.logs.debug_log('fit_success = {}'.format(fit_success))    
    
    model.set_R(model.R) # conservative reduction in R.
    hm.logs.debug_log('model.R = {}'.format(model.R))
    
    #===========================================================================
    # Computing evidence using learnt model and emcee chains
    #===========================================================================
    hm.logs.critical_log('Compute evidence...')
    hm.logs.debug_log('---------------------------------')
    """
    Instantiates the evidence class with a given model. Adds some chains and 
    computes the log-space evidence (marginal likelihood).
    """
    ev = hm.Evidence(chains_test.nchains, model)
    ev.add_chains(chains_test)
    ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
    clock = time.clock() - clock
    hm.logs.critical_log('execution_time = {}s'.format(clock))
    
    
    #===========================================================================
    # Display evidence results 
    #===========================================================================
    hm.logs.debug_log('---------------------------------')
    hm.logs.critical_log('Log-space Statistics')
    hm.logs.debug_log('---------------------------------')
    hm.logs.debug_log('ln( z ) = {}'
        .format(ev.ln_evidence_inv))
    hm.logs.debug_log('ln( std(z) ) = {}, ln ( std(z)/z ) = {}'
        .format(0.5*ev.ln_evidence_inv_var, \
                0.5 * ev.ln_evidence_inv_var - ev.ln_evidence_inv))
    hm.logs.debug_log('ln( kurt ) = {}, sqrt( 2/(n_eff-1) ) = {}'
        .format(ev.ln_kurtosis, np.sqrt(2.0/(ev.n_eff-1))))    
    hm.logs.debug_log('ln( std(var(z))/var(z) ) = {}'
        .format(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var) )
    hm.logs.debug_log('---------------------------------')
    hm.logs.critical_log('Real-space Statistics')
    hm.logs.debug_log('---------------------------------')
    hm.logs.debug_log('exp(ln( z )) = {}'
        .format(np.exp(ev.ln_evidence_inv) ) )
    hm.logs.debug_log('exp(ln( std(z) ))= {}, exp(ln( std(z)/z )) = {}'
        .format(np.exp( 0.5*ev.ln_evidence_inv_var), \
                np.exp(0.5 * ev.ln_evidence_inv_var - ev.ln_evidence_inv)) )
    hm.logs.debug_log('exp(ln( kurt )) = {}, sqrt( 2/(n_eff-1) ) = {}'
        .format(np.exp( ev.ln_kurtosis ), np.sqrt(2.0/(ev.n_eff-1))))    
    hm.logs.debug_log('exp(ln( std(var(z))/var(z)))) = {}'
        .format(np.exp( 0.5 * ev.ln_evidence_inv_var_var \
                            - ev.ln_evidence_inv_var) ) )
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

    #===========================================================================
    # Plotting and prediction functions
    #===========================================================================
    
    
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
    
    # Define parameters.
    model_1=False
    nchains = 400
    # samples_per_chain = 1000000
    samples_per_chain = 20000
    nburn = 2000
    np.random.seed(2)
    
    # Run example.
    samples = run_example(model_1, nchains, samples_per_chain, nburn, 
                          plot_corner=True, plot_surface=True,
                          plot_comparison=True, 
                          verbose=True)

