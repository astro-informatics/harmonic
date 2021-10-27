A modified Gaussian mixture model provides greater flexibility that the simple hypersphere model. In particular, it is much more effective for multimodal posterior distributions.

Consider the target distribution defined by the modified Gaussian mixture model

.. math:: \varphi(\theta) = \sum_{k=1}^K \frac{w_k}{(2\pi)^{d/2} \vert \Sigma_k \vert^{1/2} s_k^d} \exp \biggl( \frac{- \bigl(\theta - \bar{\theta}_k\bigr)^\text{T} \Sigma_k^{-1} \bigl(\theta - \bar{\theta}_k\bigr)}{2 s_k^2}\biggr),

for :math:`K` components, with centres :math:`\bar{\theta}_k` and covariances :math:`\Sigma_k^{-1}`, where the relative scale of each component is controlled by :math:`s_k` and the weights are specified by

.. math:: w_k = \frac{\exp(z_k)}{\sum_{k^\prime=1}^K \exp(z_{k^\prime})}.

Given :math:`K`, the posterior training samples can be clustered by :math:`K`-means.  The values of :math:`\bar{\theta}_k` and :math:`\Sigma_k` can then be computed by the samples in cluster :math:`k`.  The model is modified relative to the usual Gaussian mixture model in that the cluster mean and covariance are estimated from the samples of each cluster, while the relative cluster scale and weights are fitted.  Moreover, as before, a bespoke training approach is adopted tailored to the problem of learning an effective model for the learnt harmonic mean estimator.

To estimate the the weights :math:`z_k`, which in turn define the weights :math:`w_k`, and the relative scales :math:`s_k` we again set up an optimisation problem to minimise the variance of the learnt harmonic mean estimator, while also constraining it to be unbiased.  We also regularise the relative scale parameters, resulting in the following optimisation problem:

.. math:: \min_{\{z_k,s_k\}_{k=1}^K} \: \hat{\sigma}^2 + \frac{1}{2} \lambda \sum_{k=1}^K s_k^2 \quad \text{s.t.} \quad \hat{\rho} = \hat{\mu}_1,

for regularisation parameter :math:`\lambda`. The problem may equivalently be written as

.. math:: \min_{\{z_k,s_k\}_{k=1}^K} \: \hat{\mu}_2 + \frac{1}{2} \lambda \sum_{k=1}^K s_k^2,

or explicitly in terms of the posterior samples by

.. math:: \min_{\{z_k,s_k\}_{k=1}^K} \: \sum_i C_i^2 + \frac{1}{2} \lambda \sum_{k=1}^K s_k^2.

The individual cost terms for each sample :math:`i` are given by

.. math:: C_i = \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i) \pi(\theta_i)} = \sum_{k=1}^K C_{ik},

which include the following component from cluster :math:`k`:

.. math:: C_{ik} = \frac{w_k}{(2\pi)^{d/2} \vert \Sigma_k \vert^{1/2} s_k^d}\, \exp \biggl(\frac{- \bigl(\theta_i - \bar{\theta}_k\bigr)^\text{T} \Sigma_k^{-1} \bigl(\theta_i - \bar{\theta}_k\bigr)}{2 s_k^2}\biggr)\frac{1}{\mathcal{L}(\theta_i) \pi(\theta_i)}.

We solve this optimisation problem by stochastic gradient decent, which requires the gradients of the objective function. Denoting the total cost of the objective function by :math:`C = \sum_i C_i^2 + \frac{1}{2} \lambda \sum_{k=1}^K s_k^2`, it is straightforward to show that the gradients of the cost function with respective to the weightsgularise th :math:`z_k` and relative scales :math:`s_k` are given by

.. math:: \frac{\partial C}{\partial z_k} = 2 \sum_i C_i (C_{ik} - w_k C_i),

and

.. math:: \frac{\partial C}{\partial s_k} = 2 \sum_i \frac{C_i C_{ik}}{s_k^3}\Bigl( \bigl(\theta_i - \bar{\theta}_k\bigr)^\text{T} \Sigma_k^{-1} \bigl(\theta_i - \bar{\theta}_k\bigr) - d s_k^2 \Bigr),

respectively.

The general procedure to learn the target distribution is the same as before: first, construct a normalised model; second, train the model by solving an optimisation problem to minimise the variance of the resulting learnt harmonic mean estimator. In this case we regularise the relative scale parameters and then solve by stochastic gradient descent. The number of clusters :math:`K` can be deteremined by cross-validation.

While the modified Gaussian mixture model can effectively handle multimodal distributions, alternative models are better suited to narrow curving posterior degeneracies.
