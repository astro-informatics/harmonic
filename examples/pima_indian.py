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


def ln_likelihood(y, theta, x):
    """Compute log_e of Pima Indian likelihood.

    Args:

        y: Vector of diabetes incidence (1=diabetes, 0=no diabetes).

        theta: Vector of parameter variables associated with covariates x.

        x: Vector of data covariates (e.g. NP, PGC, BP, TST, DMI etc.).

    Returns:

        double: Value of log_e likelihood at specified point in parameter
            space.

    """

    ln_p = compute_ln_p(theta, x)
    ln_pp = np.log(1. - np.exp(ln_p))
    return y.T.dot(ln_p) + (1-y).T.dot(ln_pp)


def ln_prior(tau, theta): 
    """Compute log_e of Pima Indian multivariate gaussian prior.

    Args:

        tau: Characteristic width of posterior \in [0.01,1].

        theta: Vector of parameter variables associated with covariates x.

    Returns:

        double: Value of log_e prior at specified point in parameter space.

    """

    d = len(theta)
    return 0.5 * d * np.log(tau/(2.*np.pi)) - 0.5 * tau * theta.T.dot(theta)


def ln_posterior(theta, tau, x, y): 
    """Compute log_e of Pima Indian multivariate gaussian prior

    Args:

        theta: Vector of parameter variables associated with covariates x.

        tau: Characteristic width of posterior \in [0.01,1].

        x: Vector of data covariates (e.g. NP, PGC, BP, TST, DMI etc).

        y: Vector of incidence. 1=diabetes, 0=no diabetes.

    Returns:

        double: Value of log_e posterior at specified point in parameter
            space.

    """

    ln_pr = ln_prior(tau, theta)
    ln_L = ln_likelihood(y, theta, x)

    return ln_pr + ln_L


def compute_ln_p(theta, x):
    """Computes log_e probability ln(p) to be used in likelihood function.

    Args:

        theta: Vector of parameter variables associated with covariates x.

        x: Vector of data covariates (e.g. NP, PGC, BP, TST, DMI e.t.c.).

    Returns:

        double: Vector of the log-probabilities p to use in likelihood.

    """

    return - np.log(1.0 + 1.0 / np.exp(x.dot(theta)))


def run_example(model_1=True, tau=1.0,
                nchains=100, samples_per_chain=1000,
                nburn=500, verbose=True, 
                plot_corner=False, plot_surface=False):
    """Run Pima Indians example.

    Args:

        model_1: Consider model 1 if true, otherwise model 2.

        tau: Precision parameter.

        nchains: Number of chains.

        samples_per_chain: Number of samples per chain.

        nburn: Number of burn in samples for each chain.

        verbose: If True then display intermediate results.

        plot_corner: Plot marginalised distributions if true.

        plot_surface: Plot surface and samples if true.

    """

    hm.logs.debug_log('---------------------------------')
    hm.logs.critical_log('Pima Indian example')

    # Set_dimension
    if model_1:
        ndim = 5
    else:
        ndim = 6
    hm.logs.critical_log('Dimensionality = {}'.format(ndim))
    hm.logs.debug_log('---------------------------------')

    #===========================================================================
    # Load Pima Indian data.
    #===========================================================================
    hm.logs.critical_log('Loading data ...')
    hm.logs.debug_log('---------------------------------')

    data = np.loadtxt('examples/data/PimaIndian.dat')

    """
    Two primary models for comparison:
        Model 1: Uses rows const(1) + data(1,2,5,6) = 5 dimensional
        Model 2: Uses rows const(1) + data(1,2,5,6,7) = 6 dimensional
    data[:,0] --> Diabetes incidence. 
    data[:,1] --> Number of pregnancies (NP)
    data[:,2] --> Plasma glucose concentration (PGC)
    data[:,3] --> Diastolic blood pressure (BP)
    data[:,4] --> Tricept skin fold thickness (TST)
    data[:,5] --> Body mass index (BMI)
    data[:,6] --> Diabetes pedigree function (DP)
    data[:,7] --> Age (AGE)
    """
    x=np.zeros((len(data), ndim))

    if model_1:
        x[:,0] = 1.0
        x[:,1] = data[:,1]
        x[:,2] = data[:,2]
        x[:,3] = data[:,5]
        x[:,4] = data[:,6]

    else:
        x[:,0] = 1.0
        x[:,1] = data[:,1]
        x[:,2] = data[:,2]
        x[:,3] = data[:,5]
        x[:,4] = data[:,6]
        x[:,5] = data[:,7] # --> model 2.

    """
    y[:] = 1 if patient has diabetes, 0 if patient does not have diabetes.
    """
    y = data[:,0]

    """
    Configure some general parameters.
    """
    savefigs = True

    """
    Configure machine learning parameters.
    """
    nfold = 3
    training_proportion = 0.25
    hyper_parameters_MGMM = [[1, 1E-8, 0.1, 6, 10], [2, 1E-8, 0.5, 6, 10]]
    hyper_parameters_sphere = [None]
    domains_sphere = [np.array([1E-2,1E0])]
    domains_MGMM = [np.array([1E-2,1E0])]

    #===========================================================================
    # Compute random positions to draw from for emcee sampler.
    #===========================================================================
    """
    Initial positions for each chain for each covariate \in [0,8).
    Simply drawn from directly from each covariate prior.
    """
    if model_1:
        pos_0 = np.random.randn(nchains)*0.01
        pos_1 = np.random.randn(nchains)*0.01
        pos_2 = np.random.randn(nchains)*0.01
        pos_3 = np.random.randn(nchains)*0.01
        pos_4 = np.random.randn(nchains)*0.01
        pos = np.c_[pos_0, pos_1, pos_2, pos_3, pos_4]

    else:
        pos_0 = np.random.randn(nchains)*0.01
        pos_1 = np.random.randn(nchains)*0.01
        pos_2 = np.random.randn(nchains)*0.01
        pos_3 = np.random.randn(nchains)*0.01
        pos_4 = np.random.randn(nchains)*0.01
        pos_5 = np.random.randn(nchains)*0.01 
        pos = np.c_[pos_0, pos_1, pos_2, pos_3, pos_4, pos_5]

    # Start Timer.
    clock = time.clock()

    #===========================================================================
    # Run Emcee to recover posterior sampels 
    #===========================================================================
    hm.logs.critical_log('Run sampling...')
    hm.logs.debug_log('---------------------------------')
    """
    Feed emcee the ln_posterior function, starting positions and recover chains.
    """
    sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, \
                                    args=(tau, x, y))
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
    # Perform cross-validation
    #===========================================================================
    hm.logs.critical_log('Perform cross-validation...')
    hm.logs.debug_log('---------------------------------')
    """
    There are several different machine learning models. Cross-validation
    allows the software to select the optimal model and the optimal model 
    hyper-parameters for a given situation.
    """
    # MGMM model
    validation_variances_MGMM = \
        hm.utils.cross_validation(chains_train, 
            domains_MGMM, \
            hyper_parameters_MGMM, \
            nfold=nfold, 
            modelClass=hm.model.ModifiedGaussianMixtureModel, \
            verbose=verbose, seed=0)                
    hm.logs.debug_log('Validation variances of MGMM = {}'
        .format(validation_variances_MGMM))
    best_hyper_param_MGMM_ind = np.argmin(validation_variances_MGMM)
    best_hyper_param_MGMM = \
        hyper_parameters_MGMM[best_hyper_param_MGMM_ind]

    # Hyper-spherical model
    validation_variances_sphere = \
        hm.utils.cross_validation(chains_train, 
            domains_sphere, \
            hyper_parameters_sphere, nfold=nfold, 
            modelClass=hm.model.HyperSphere, 
            verbose=verbose, seed=0)
    hm.logs.debug_log('Validation variances of sphere = {}'
        .format(validation_variances_sphere))
    best_hyper_param_sphere_ind = np.argmin(validation_variances_sphere)
    best_hyper_param_sphere = \
        hyper_parameters_sphere[best_hyper_param_sphere_ind]

    #===========================================================================
    # Select the optimal model from cross-validation results
    #===========================================================================
    hm.logs.critical_log('Select optimal model...')
    hm.logs.debug_log('---------------------------------')
    """
    This simply uses the cross-validation results to choose the model which 
    has the smallest validation variance -- i.e. the best model for the job.
    """
    best_var_MGMM = \
        validation_variances_MGMM[best_hyper_param_MGMM_ind]
    best_var_sphere = \
        validation_variances_sphere[best_hyper_param_sphere_ind]
    if best_var_MGMM < best_var_sphere:            
        hm.logs.critical_log('Using MGMM with hyper_parameters = {}'
            .format(best_hyper_param_MGMM))                
        model = hm.model.ModifiedGaussianMixtureModel(ndim, \
            domains_MGMM, hyper_parameters=best_hyper_param_MGMM)
        model.verbose=False
    else:
        hm.logs.critical_log('Using HyperSphere')
        model = hm.model.HyperSphere(ndim, domains_sphere, \
                hyper_parameters=best_hyper_param_sphere)
        model = hm.model.HyperSphere(ndim, domains_sphere,hyper_parameters=None)            

    #===========================================================================
    # Fit learnt model for container function 
    #===========================================================================
    """
    Once the model is selected the model is fit to chain samples.
    """
    fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)
    hm.logs.debug_log('Fit success = {}'.format(fit_success)) 

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
    evidence_std_log_space = np.log(np.exp(ln_evidence) + np.exp(ln_evidence_std)) - ln_evidence

    #===========================================================================
    # End Timer.
    clock = time.clock() - clock
    hm.logs.critical_log('Execution time = {}s'.format(clock))

    #===========================================================================
    # Display evidence results 
    #===========================================================================
    hm.logs.critical_log('---------------------------------')
    hm.logs.critical_log('Results')
    hm.logs.critical_log('---------------------------------')
    hm.logs.critical_log('ln_evidence = {} +/- {}'.format(ln_evidence, evidence_std_log_space))
    hm.logs.critical_log('kurtosis = {}'.format(ev.kurtosis))
    hm.logs.critical_log('sqrt( 2/(n_eff-1) ) = {}'.format(np.sqrt(2.0/(ev.n_eff-1))))
    check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
    hm.logs.critical_log('sqrt(evidence_inv_var_var) / evidence_inv_var = {}'.format(check))

    #===========================================================================
    # Display more technical details
    #===========================================================================
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
    hm.logs.debug_log('running sum total = {}'
        .format(sum(ev.running_sum)))
    hm.logs.debug_log('running sum = \n{}'
        .format(ev.running_sum))
    hm.logs.debug_log('nsamples per chain = \n{}'
        .format(ev.nsamples_per_chain))
    hm.logs.debug_log('nsamples eff per chain = \n{}'
        .format(ev.nsamples_eff_per_chain))
    hm.logs.debug_log('===============================')


if __name__ == '__main__':

    # Setup logging config.
    hm.logs.setup_logging()

    # Define problem parameters
    model_1 = False
    # Tau should be varied in [0.01, 1].
    #tau = 1.0
    tau = 0.01

    # Define parameters.
    nchains = 200
    samples_per_chain = 5000
    nburn = 1000
    np.random.seed(3)
    
    # Run example.
    samples = run_example(model_1, tau, nchains, samples_per_chain, nburn, 
                          plot_corner=False, plot_surface=False,
                          verbose=False)
