*************************
Outputs from harmonic
*************************

**harmonic** recovers estimators for :math:`z` and :math:`\text{Var}[z]`, denoted by :math:`\hat{z}` and :math:`\text{Var}[\hat{z}]` respectively. In high-dimensional settings these values often are in danger of over- (or under-) flowing, particularly at 32 bit numerical precision. To avoid this practitioners often consider instead estimators for :math:`\log \big ( z \big )` upon which they recover estimates of, *e.g.* :math:`\text{Var}[\log \big ( z \big )]` *etc.*

The harmonic mean estimator by construction provides estimators of :math:`z` and is thus susceptible to these over- (or under-) flowing concerns. In **harmonic** we leverage numerical tricks, such as the `LogSumExp trick <https://en.wikipedia.org/wiki/LogSumExp>`_, to mitigate this, however this returns estimators for :math:`\log \big ( \hat{z} \big )` and :math:`\log \big ( \text{Var}[\hat{z}] \big )` respectively, which are **not** equivalent to their log-space counterparts. This can be straightforwardly seen by considering :math:`\log \big ( \text{Var}[\hat{z}] \big ) \not = \text{Var}[\hat{\log \big (z \big )}]`, which follows as summation and the natural logarithm do not commute.

Given values for :math:`\log \big ( \hat{z} \big )` and :math:`\log \big ( \text{Var}[\hat{z}] \big )`, and in the situation wherein computing their exponents will immediately over- (or under-) flow, we can get a notion of how :math:`\hat{\sigma}` manifests on :math:`\hat{z}` in log-space by considering the limiting terms :math:`x_{\pm} = \log \big (\hat{z} \pm \hat{\sigma} \big )`. One way to do this is by simply computing

.. math::

   x_{\pm} = \log \big (\hat{z} \pm \hat{\sigma} \big ) 
     = \log \Big (\hat{z} \big ( 1 \pm \frac{\hat{\sigma}}{\hat{z}} \big ) \Big ) 
     = \log \big (\hat{z} \big ) + \log \big ( 1 \pm \frac{\hat{\sigma}}{\hat{z}} \big ),

where we now have :math:`\eta = \hat{\sigma}/\hat{z}` which may, at first, appear to be potentially problematic. However, we can compute this ratio as a distance in log-space, normalising away any intractably large numbers, which is then almost certainly well-behaved,

.. math::

  \eta = \frac{\hat{\sigma}}{\hat{z}} = \exp \bigg ( 
  \underbrace{ 
    \log \big ( \hat{\sigma} \big ) - \log \big ( \hat{z} \big )
  }_{\text{At most $\sim \mathcal{O}(8)$}} 
  \bigg ).

Hence, :math:`x_{\pm} = \log \big ( \hat{z} \pm \hat{\sigma} \big )` is straightforward to calculate, provided the log of the estimated variance isn't further than :math:`\sim 88` from the log of the estimate of the evidence (the over/under-flow limit of the exponential function at 32 bit precision).