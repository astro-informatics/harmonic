.. _logvar:

******************
Log-space variance
******************

``harmonic`` computes estimators of the reciprocal evidence and its variance, denoted by :math:`\hat{\rho}` and :math:`\hat{\sigma}^2` respectively. 

We compute the natural logarithm of these quantities for numerical stability (using a number of numerical tricks, such as the `LogSumExp trick <https://en.wikipedia.org/wiki/LogSumExp>`_).  We therefore directly compute :math:`\log ( \hat{\rho} )` and :math:`\log ( \hat{\sigma} )`.  

While quantities are computed in log space for numerical stability, the variance that is computed relates to the underlying reciprocal evidence (**not** the log reciprocal evidence), i.e. the terms computed can be considered as the estimate and error :math:`\hat{\rho} \pm \hat{\sigma}`.  This is of course not equivalent to a variance estimate of the log space estimate, which can be seen straightforwardly since :math:`\log ( \text{var}(x) ) \not = \text{var}(\log (x ))` as summation and the natural logarithm do not commute.

In some settings one may be interested in an error estimate defined in log space, i.e. the log-space error :math:`\hat{\zeta}_\pm` defined by

.. math::

  \log ( \hat{\rho} \pm \hat{\sigma} ) = \log (\hat{\rho}) + \hat{\zeta}_\pm .

The log-space error estimate can be computed by

.. math::

  \hat{\zeta}_\pm = \log (\hat{\rho} \pm \hat{\sigma} ) - \log (\hat{\rho}) .

This may also be expressed as 

.. math::

  \hat{\zeta}_\pm = \log(1 \pm \hat{\sigma} / \hat{\rho}) ,

where 

.. math::

  \hat{\sigma} / \hat{\rho} =  \exp \bigl( \log(\hat{\sigma}) - \log(\hat{\rho}) \bigr) .

The advantage of this second approach is that it avoids explicitly computing :math:`\exp(\log(\hat{\rho})) \pm \exp(\log(\hat{\sigma}))`, which has the potental to over- or under-flow.  The second, more stable, approach is implemented in :code:`Evidence.compute_ln_inv_evidence_errors`.
