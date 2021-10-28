We consider the comparison of two logistic regression models using the **Pima Indians** data, which is another common benchmark problem for comparing estimators of the marginal likelihood.  The original harmonic mean estimator has been shown to fail catastrophically for this example, whereas we show here that our learnt harmonic mean estimator is highly accurate.

The Pima Indians data, originally from the National Institute of Diabetes and Digestive and Kidney Diseases, were compiled from a study of indicators of diabetes in :math:`n=532` Pima Indian women aged 21 or over.  Seven primary predictors of diabetes were recorded, including: number of prior pregnancies (NP);  plasma glucose concentration (PGC); diastolic blood pressure (BP); triceps skin fold thickness (TST); body mass index (BMI); diabetes pedigree function (DP); and age (AGE).

The probability of diabetes :math:`p_i` for person :math:`i \in \{1, \ldots, n\}` can be modelled by the standard logistic function

.. math:: p_i = \frac{1}{1+\exp\bigl(- \theta^\text{T} x_i\bigr)},

with covariates :math:`x_i = (1,x_{i,1}, \dots x_{i,d})^\text{T}` and parameters :math:`\theta = (\theta_0, \dots, \theta_d)^\text{T}`, where :math:`d` is the total number of covariates considered.  The likelihood function then reads

.. math:: \mathcal{L}({y} | {\theta}) = \prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i},

where :math:`y = (y_1, \dots, y_n)^\text{T}` is the diabetes incidence, (*i.e.* :math:`y_i` is unity if patient :math:`i` had diabetes and zero otherwise). An independent multivariate Gaussian prior is assumed for the parameters :math:`\theta`, given by

.. math:: \pi(\theta) = \Bigl(  \frac{\tau}{2\pi} \Bigr)^{d/2} \exp \bigl( - \frac{\tau}{2} \theta^\text{T} \theta \bigr),

with precision :math:`\tau`. Two different logistic regression models are considered, with different subsets of covariates:

.. math::
  :nowrap:

    \begin{align}
    &\text{Model 1: covariates = \{NP, PGC, BMI, DP\} (and bias),} \\
    &\text{Model 2: covariates = \{NP, PGC, BMI, DP, AGE\} (and bias)}.
    \end{align}

A graphical representation of Model 2 is illustrated below (Model 1 is similar but does not include the AGE covariate).

.. image:: /assets/dags/hbm_pima_indian.svg
	:width: 50 %
	:align: center

The log-likelihood function is given by

.. code-block:: python

   def ln_likelihood(y, theta, x):

    ln_p = compute_ln_p(theta, x)
    ln_pp = np.log(1. - np.exp(ln_p))

    return y.T.dot(ln_p) + (1-y).T.dot(ln_pp)

The log-prior is given by a multivariate Gaussian, *e.g.*

.. code-block:: python

   def ln_prior(tau, theta): 

    return 0.5 * len(theta) * np.log(tau/(2.*np.pi)) - 0.5 * tau * theta.T.dot(theta)

We may then combine the log-likelihood and log-prior functions to define the log-posterior function simply by

.. code-block:: python
	
   def ln_posterior(theta, tau, x, y): 

    ln_pr = ln_prior(tau, theta)
    ln_L = ln_likelihood(y, theta, x)

    return ln_pr + ln_L

The first step of our evidence computation requires recovering a relatively small number of samples from the given posterior. This can be done in whatever way the user wishes, the only requirement being that a set of chains each with associated samples is provided for subsequent steps.
In our examples we choose to use the excellent `emcee  <http://dfm.io/emcee/current/>`_ python package. Utilizing emcee this example recovers samples via 

.. code-block:: python
	
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

   sampler = emcee.EnsembleSampler(nchains, ndim, ln_posterior, args=(tau, x, y))
   rstate = np.random.get_state()
   sampler.run_mcmc(pos, samples_per_chain, rstate0=rstate)
   samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
   lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

where the initial positions are drawn randomly from the support of each covariate prior.

Cross-Validation 
==========================
The cross-validation step allows **Harmonic** to compute the optimal hyper-parameter configuration for a certain class of model for a given set of posterior samples. There are two main stages to this cross-validation process. First the MCMC chains (in this case from emcee) are configured

.. code-block:: python

   chains = hm.Chains(ndim)
   chains.add_chains_3d(samples, lnprob)
   chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.5)

before being used as training data to train a network to predict optimal configurations of the hyper-parameters associated with the model class. This is done by

.. code-block:: python

   # MGMM model
    validation_variances_MGMM = 
        hm.utils.cross_validation(chains_train, 
            domains_MGMM, 
            hyper_parameters_MGMM, 
            nfold=nfold, 
            modelClass=hm.model.ModifiedGaussianMixtureModel, 
            seed=0)                
    best_hyper_param_MGMM_ind = np.argmin(validation_variances_MGMM)
    best_hyper_param_MGMM = hyper_parameters_MGMM[best_hyper_param_MGMM_ind]

    # Hyper-spherical model
    validation_variances_sphere = 
        hm.utils.cross_validation(chains_train, 
            domains_sphere, 
            hyper_parameters_sphere, nfold=nfold, 
            modelClass=hm.model.HyperSphere, 
            seed=0)
    best_hyper_param_sphere_ind = np.argmin(validation_variances_sphere)
    best_hyper_param_sphere = hyper_parameters_sphere[best_hyper_param_sphere_ind]

In this case we adopt cross-validation to select between the MGMM and Hyper-spherical models, as it is not necessarily clear which is more effective. The most effective model is selected by 

.. code-block:: python

   best_var_MGMM = validation_variances_MGMM[best_hyper_param_MGMM_ind]
   best_var_sphere = validation_variances_sphere[best_hyper_param_sphere_ind]
   if best_var_MGMM < best_var_sphere:                           
        model = hm.model.ModifiedGaussianMixtureModel(
        ndim, domains_MGMM, hyper_parameters=best_hyper_param_MGMM)
   else:
        model = hm.model.HyperSphere(
        ndim, domains_sphere, hyper_parameters=best_hyper_param_sphere)   

Finally the now sucessfully trained network is used to make a prediction (fit) the optimal (learnt) container function :math:`\psi` -- *i.e.* the optimal hyper-parameter configuration -- by

.. code-block:: python

   fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)

This learnt container function is then used with the harmonic mean estimator to construct a robust computation of the Bayesian evidence by

.. code-block:: python

   ev = hm.Evidence(chains_test.nchains, model)    
   ev.add_chains(chains_test)
   ln_evidence, ln_evidence_std = ev.compute_ln_evidence()