**************
Bayes factors
**************

We have so far considered the estimation of the reciprocal marginal likelihood and related variances only.  However, it is the marginal likelihood itself (not its reciprocal), or the Bayes factors computed to compare two models, that is typically of direct interest.  We therefore consider how to compute these quantities of interest and a measure of their variance (as discussed in `McEwen et al. 2021 <https://arxiv.org/abs/2111.12720>`_).

First, consider the mean and variance of the function :math:`f(X,Y) = X/Y` of two uncorrelated random variables :math:`X` and :math:`Y`, which by Taylor expansion to second order are given by

.. math::

  \mathbb{E}\biggl(\frac{X}{Y}\biggr) \simeq \frac{\mathbb{E}(X)}{\mathbb{E}(Y)} + \frac{\mathbb{E}(X)}{\mathbb{E}(Y)^3} \sigma_Y^2
  
and

.. math::

  \text{var}\biggl(\frac{X}{Y}\biggr) \simeq \frac{1}{\mathbb{E}(Y)^2} \sigma_X^2 + \frac{\mathbb{E}(X)^2}{\mathbb{E}(Y)^4} \sigma_Y^2,

respectively, where :math:`\sigma_X = \mathbb{E}\bigl[ (X - \mathbb{E}(X))^2 \bigr]` and :math:`\sigma_Y = \mathbb{E}\bigl[ (Y - \mathbb{E}(Y))^2 \bigr]`.

Using this result the marginal likelihood and its variance can be estimated from the reciprocal estimators by making use of the relations

.. math:: 

  \mathbb{E}( z )
  = \mathbb{E}\biggl(\frac{1}{{\rho}}\biggr)
  \simeq \frac{1}{\mathbb{E}({\rho})} \biggl( 1 + \frac{\sigma_{\rho}^2}{\mathbb{E}({\rho})^2} \biggr)  
  \qquad \quad

and 

.. math:: 

  \text{var}(z) =
  \text{var}\biggl(\frac{1}{{\rho}}\biggr)
  \simeq \frac{\sigma_{\rho}^2}{\mathbb{E}({\rho})^4},

respectively, by considering the case :math:`X=1` and :math:`Y = \rho`.  These expressions are computed (in log space) in ``harmonic`` by the :code:`Evidence.compute_ln_evidence` method.

Typically it is the Bayes factor given by the ratio of marginal likelihoods that is of most interest in order to compare models.  Again using the expressions above for the mean and variance of the function :math:`f(X,Y) = X/Y`, this time for the case :math:`X = \rho_2` and :math:`Y = \rho_1`, the Bayes factor and its variance can be estimated directly from the reciprocal marginal likelihood estimates and variances by making use of the relations

.. math::

  \mathbb{E}\biggl(\frac{z_1}{z_2}\biggr)
  =
  \mathbb{E}\biggr(\frac{\rho_2}{\rho_1}\biggr)
  \simeq
  \frac{\mathbb{E}({\rho_2})}{\mathbb{E}({\rho_1})}
  \biggl( 1 + \frac{\sigma_{\rho_1}^2}{\mathbb{E}({\rho_1})^2}  \biggr)

and

.. math::

  \text{var}\biggl(\frac{z_1}{z_2}\biggr)
  =
  \text{var}\biggl(\frac{{\rho_2}}{{\rho_1}}\biggr)
  \simeq
  \frac{\mathbb{E}({\rho_1})^2 \sigma_{\rho_2}^2+ \mathbb{E}({\rho_2})^2 \sigma_{\rho_1}^2}{\mathbb{E}({\rho_1})^4},

respectively.  These expressions are computed (in log space) in ``harmonic`` by the :code:`Evidence.compute_ln_bayes_factor` function.

While we compute log quantities, as discussed above, the variance computed (i.e. error estimate) is related to the evidence itself (not the log evidence).  We consider how to compute an error estimate for the log evidence in the section :ref:`logvar`.
