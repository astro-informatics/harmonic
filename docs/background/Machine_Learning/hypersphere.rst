The simplest model one may wish to consider is a hypersphere, much like the truncated harmonic mean estimator. However, here we learn the optimal radius of the hypersphere.

Consider the target distribution defined by the normalised hypersphere

.. math:: \varphi(\theta) = \frac{1}{V_\mathcal{S}}  I_\mathcal{S}(\theta),

where the indicator function :math:`I_\mathcal{S}(\theta)` is unity if :math:`\theta` is within a hypersphere of radius :math:`R`, centred on :math:`\bar{\theta}` with covariance :math:`\Sigma`, *i.e*.

.. math:: I_\mathcal{S}(\theta) =
            \begin{cases}
              1, & \bigl(\theta - \bar{\theta}\bigr)^\text{T} \Sigma^{-1} \bigl(\theta - \bar{\theta} \bigr) < R^2 \\
              0, & \text{otherwise}
            \end{cases}.

The values of :math:`\bar{\theta}` and :math:`\Sigma` can be computed directly from the training samples.  Often a diagonal approximation of :math:`\Sigma` is considered for computational efficiency. The volume of the hypersphere required to normalise the distribution is given by

.. math:: V_\mathcal{S} = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)} R^d \, \vert \Sigma \vert^{1/2}.

Recall that :math:`d` is the dimension of the parameter space, *i.e.* :math:`\theta \in \mathbb{R}^d`, and note that :math:`\Gamma(\cdot)` is the Gamma function. To estimate the radius of the hypersphere we pose the following optimisation problem to minimise the variance of the learnt harmonic mean estimator, while also constraining it be be unbiased:

.. math:: \min_R \: \hat{\sigma}^2 \quad \text{s.t.} \quad \hat{\rho} = \hat{\mu}_1.

By minimising the variance of the estimator we ensure, on one hand, that the tails of the learnt target are not so wide that they are broader than the posterior, and, on the other hand, that they are not so narrow that very few samples are effectively retained in the estimator.

This optimisation problem is equivalent to minimising the estimator of the second harmonic moment:

.. math:: \min_R \: \hat{\mu}_2.

Writing out the cost function explicitly in terms of the posterior samples, the optimisation problem reads 

.. math:: \min_R \: \sum_i C_i^2, 

with costs for each sample given by

.. math:: C_i = \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)} \propto
              \begin{cases}
                \frac{1}{\mathcal{L}(\theta_i) \pi(\theta_i) R^d}, & \bigl(\theta - \bar{\theta}\bigr)^\text{T} \Sigma^{-1} \bigl(\theta - \bar{\theta} \bigr) < R^2 \\
                0,                                                 & \text{otherwise}
              \end{cases}.

This one-dimensional optimisation problem can be solved by straightforward techniques, such as the `Brent hybrid <https://en.wikipedia.org/wiki/Brent%27s_method>`_ root-finding algorithm.

While the learnt hypersphere model is very simple, it is good pedagogical illustration of the general procedure for learning target distributions. First, construct a normalised model.  Second, train the model to learn its parameters by solving an optimisation problem to minimise the variance of the estimator while ensuring it is unbiased. If required, set hyperparameters or compare alternative models by cross-validation.

While the simple learnt hypersphere model may be sufficient in some settings, it is not effective for multimodal posterior distributions or for posteriors with narrow curving degeneracies. For such scenarios we consider alternative learnt models.