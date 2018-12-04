***********************
Harmonic mean estimator
***********************
Here we provide the mathematical background of the harmonic mean estimator, 
deriving expressions for: the expectation value of the evidence :math:`\rho`, 
the variance of this estimator :math:`\text{Var}(\rho)`, and the variance in the
variance of this estimator :math:`\text{Var(Var}(\rho))`.

One can construct an estimator for the inverse Bayesian evidence (often 
referred to as the *marginal likelihood*) by considering the harmonic mean of the
likelihood over the posterior. This is to say,

.. math:: 

	\rho = \frac{1}{z} = \int d\theta \frac{\varphi(\theta)}{\mathcal{L}(D|\theta)\pi(\theta)}\frac{\mathcal{L}(D|\theta)\pi(\theta)}{z},

where :math:`\varphi(\theta)` is an arbitrary **container function** 
which one is free to choose provided that the function is *proper*. See the 
machine learning section for details on how one should select 
:math:`\varphi(\theta)`.


Uncorrelated samples
====================

This subsection will consider estimating the inverse Bayesian evidence from 
**independant samples** drawn directly from the posterior distribution. To do so 
we will first need to define the :math:`n^{\text{th}}` harmonic moment of the 
posterior to be

.. math:: 

	\mu_n =\mathbb{E}\left[\left(\frac{\varphi(\theta)}{\mathcal{L}(\theta)\pi(\theta)}\right)^n\right]_{P(\theta)},

where :math:`\mathbb{E}[.]_{p(x)}` is the expectation over the distribution 
:math:`p(x)`.



Estimate of :math:`\widehat{\rho}` and :math:`\mathbb{V}\text{ar}(\widehat{\rho})`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Suppose we have :math:`N_s \gg 1` independent posterior samples, then the 
inverse evidence estimator is given by,

.. math:: 

	\widehat{\rho} = \frac{1}{N_{\rm s}}\sum_{i=1}^{N_{\rm s}} \frac{\varphi(\theta_i)}{{\mathcal{L}(\theta_i)\pi(\theta_i)}} \quad\quad \theta_i \sim P(\theta),

where hat denotes an estimate. The expectation value is then 
simpy given by,

.. math:: 
	:nowrap:

	\begin{align}
	\mathbb{E}\left[\widehat{\rho}\right] &= \mathbb{E}\left[\frac{1}{N_{\rm s}} \sum_{i=1}^{N_{\rm s}} \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i)\pi(\theta_i)}\right], \quad\quad \theta_i \sim P(\theta), \\
	&= \frac{1}{N_{\rm s}} \sum_{i=1}^{N_{\rm s}} \mathbb{E}\left[\frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i)\pi(\theta_i)}\right],\\
	&= \mathbb{E}\left[\frac{\varphi(\theta)}{\mathcal{L}(\theta)\pi(\theta)}\right]_{P(\theta)}, \\
	&= \rho.
	\end{align}

In precisely the same way one can construct the variance in this estimator as,

.. math:: 
	:nowrap:

	\begin{align}
	\mathbb{V}\text{ar}\left[\widehat{\rho}\right] &= \mathbb{V}\text{ar}\left[\frac{1}{N_{\rm s}} \sum_{i=1}^{N_{\rm s}} \frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i)\pi(\theta_i)}\right], \quad\quad \theta_i \sim P(\theta)\\
	&= \frac{1}{N_{\rm s}} \mathbb{V}\text{ar}\left[\frac{\varphi(\theta_i)}{\mathcal{L}(\theta_i)\pi(\theta_i)}\right], \\
	&= \frac{1}{N_{\rm s}} \mathbb{V}\text{ar}\left[\frac{\varphi(\theta)}{\mathcal{L}(\theta)\pi(\theta)}\right]_{P(\theta)}, \\
	&= \frac{1}{N_{\rm s}}\left(\mu_2 - \mu_1^2\right).
	\end{align}

However this is the variance of the estimator :math:`\widehat{\rho}`, not an
estimate of the variance of :math:`\rho`.

Estimating :math:`\mathbb{V}\text{ar}(\rho)` 
++++++++++++++++++++++++++++++++++++++++++++

Typically one would like to compute the variance of the true evidence, *i.e.* 
:math:`\mathbb{V}\text{ar}(\rho)` which is given for uncorrelated samples by,

.. math::

	\widehat{\sigma^2} = \frac{1}{N_{\rm s}}\left[\frac{1}{N_{\rm s}}\sum_{i=1}^{N_{\rm s}} \left(\frac{\varphi(\theta_i)}{{\mathcal{L}(\theta_i)\pi(\theta_i)}}\right)^2 - \widehat{\rho}^2\right].  \quad\quad \theta_i \sim P(\theta).

Consider the expectation of this variance estimator,

.. math::

	\mathbb{E}\left[\widehat{\sigma^2}\right] = \frac{1}{N_{\rm s}}\left[\mathbb{E}\left[\frac{1}{N_{\rm s}}\sum_{i=1}^{N_{\rm s}}\left(\frac{\varphi(\theta_i)}{{\mathcal{L}(\theta_i)\pi(\theta_i)}}\right)^2\right] - \mathbb{E}\left[\widehat{\rho}^2\right] \right]

Note that now we can use :math:`\mathbb{E}\left[\widehat{\rho}^2\right] = \mathbb{E}\left[\widehat{\rho}\right]^2  + \mathbb{V}\text{ar}\left[\widehat{\rho}\right] = (\mu_1)^2 + \frac{1}{N_{\rm s}}\left(\mu_2 - \mu_1^2\right)` which immediately implies that

.. math::
	:nowrap:

	\begin{align}
	\mathbb{E}\left[\widehat{\sigma^2}\right] &= \frac{1}{N_{\rm s}}\left[\mu_2 -\mu_1^2 - \frac{1}{N_{\rm s}} \left(\mu_2 -\mu_1^2\right)\right],\\
	&\approx \frac{1}{N_{\rm s}}[\mu_2 - \mu_1^2],
	\end{align}


hence when a sufficiently large number of samples :math:`N_s` are used the 
estimator bias tends to zero. Using the explicit definitions of centered 
moments 

.. math::
	:nowrap:

	\begin{align}
	\mu_2^\prime &= \mu_2 - \mu_1^2,\\
	\mu_4^\prime &= \mu_4 - 4\mu_1\mu_3 + 6\mu_1^2\mu_2 - 3\mu_1^4.
	\end{align}

one can derive the variance of this variance as

.. math::
	
	\mathbb{V}\text{ar}\left[\widehat{\sigma^2}\right] \approx  \frac{1}{N^2_{\rm s}}\left[\mu_4^\prime - (\mu_2^\prime)^2\right]

which one can estimate by estimating the centered moments :math:`\mu_i` such that

.. math::
	:nowrap:

	\begin{align}
	\widehat{\nu^2} &=   \frac{1}{N^2_{\rm s}}\left[\widehat{\mu}_4^\prime - (\widehat{\mu}_2^\prime)^2\right]\\
	\widehat{\nu^2} &=   \frac{1}{N^2_{\rm s}}\left[\frac{1}{N_{\rm s}}\sum_{i=1}^{N_{\rm s}} \left(\frac{\varphi(\theta_i)}{{\mathcal{L}(\theta_i)\pi(\theta_i)}} - \widehat{\rho}\right)^4 - \left(\widehat{\sigma^2}\right)^2\right].
	\end{align}


Correlated samples
==================

Suppose now we attempt to estimate the variance in our estimator when using 
multiple chains, which are assumed to be uncorrelated (between chains) but formed
from implicitly correlated samples (within chains). It is recommended that at 
least 100 such chains are used in practice.

First, each chain is used to create an independant estimator of :math:`\rho`,

.. math::

	\widehat{p}_i = \frac{1}{N_{i}}\sum_{j=1}^{N_{i}} \frac{\varphi(\theta_j)}{{\mathcal{L}(\theta_j)\pi(\theta_j)}}, \quad\quad \theta_j \sim P(\theta)_{\rm corr},

where :math:`N_i` is the number of samples in the :math:`i^{\text{th}}` chain 
and :math:`\theta_j \sim P(\theta)_{\rm corr}` states that samples within a 
give chain are assumed to be correlated. We then perform a weighted average over 
:math:`n_c` chains 

.. math:: 
	:nowrap:

	\begin{align}
	\widehat{p} &=  \frac{\sum_{i=1}^{n_{\rm c}} w_i\widehat{p}_i}{\sum_i w_i},\\
	\widehat{s^2} &=  \frac{1}{n_{\rm eff}}\frac{\sum_{i=1}^{n_{\rm c}} w_i(\widehat{p}_i - \widehat{p})^2}{\sum_i w_i},
	\end{align}

with weights :math:`w_i = N_i`.

Understanding the estimates
+++++++++++++++++++++++++++

So we can recover the inverse evidence estimate :math:`\rho` and an estimate of 
the variance in this estimate :math:`\mathbb{V}\text{ar}(\rho)`, however to make 
statistical statements such as model selection it is necessary to consider the
variance in the variance estimator.

As one typically considers the mean of a large number of samples the central 
limit theory becomes important. As such we advocate the following approach to 
constructing an estimator for the variance of the variance.

First one computes the kurtosis, defined as 

.. math:: 

	\widehat{\mathbb{K}\text{ur}\left[\widehat{\rho_i}\right]} = \frac{\sum_{i=1}^{n_{\rm c}} w_i(\widehat{p}_i - \widehat{p})^4}{{s^4_{\rm population}}\sum_i w_i},


which is a measure of tailedness of a distribution -- for a Gaussian the kurtosis
is given by 3. Rearranging this equation we find that the variance of the 
variance can be estimated using

.. math::

	\widehat{v^2} = \frac{1}{n_{\rm eff}}\left(\widehat{s^2}\right)^2\left[\left(\widehat{\mathbb{K}\text{ur}\left[\widehat{\rho_i}\right]} -1\right) + \frac{2}{n_{\rm eff} -1}\right].

For example, if the distribution of evidence estimates :math:`\widehat{\rho}_i` 
follows a Gaussian, and one has 100 chains, the variance estimate should be 
accurate to :math:`\sim 10\%`. However :math:`\widehat{\rho}_i` need not 
necessarily follow a Gaussian distribution for this kurtosis approach to work.



Estimator for the Bayesian evidence
====================================

So far we have discussed how to construct estimators of the inverse evidence 
:math:`\widehat{\rho}` and quantities associated with it -- such as the variance
and variance of the variance -- for both correlated and uncorrelated samples.
Here we discuss briefly how one may invert these estimators to create estimators
of the Bayesian evidence (marginal likelihood).

As before we denote the evidence by :math:`z`, such that :math:`z` is given 
trivially by

.. math::
	:nowrap:

	\begin{align}
	\widehat{z} &= \widehat{\rho}^{-1},\\
	&= \rho^{-1}(1+\epsilon)^{-1},
	\end{align}

where :math:`\epsilon` is a zero mean quantity which accounts for slight 
statistical variation between :math:`\rho` and :math:`\widehat{\rho}`. The 
expectation of :math:`z` is thus given as 

.. math::
	:nowrap:

	\begin{align}
	\mathbb{E}[\widehat{z}] &= \rho^{-1}\mathbb{E}\left[(1+\epsilon)^{-1}\right],\\
	&= \rho^{-1}\mathbb{E}\left[\sum_{k=0}^\infty (-1)^k\epsilon^k\right],\\
	&= \rho^{-1}\sum_{k=0}^\infty (-1)^k\mathbb{E}\left[\epsilon^k\right],\\
	&= \rho^{-1} \left[1 + \mathbb{V}\text{ar}\left[\epsilon\right] + \mathcal{O}(\mathbb{E}\left[\epsilon^3\right])\right],\\
	&\approx \rho^{-1} \left[1 + \frac{\mathbb{V}\text{ar}\left[\widehat{\rho}\right]}{\rho^2}\right]
	\end{align}

hence the estimate is bias by the additive factor in the final line of the 
above derivation, however this can be accounted for provided one can garner 
sufficient understanding of this term. In a similar way one may construct an 
estimator for the variance of the evidence such that

.. math::

	\mathbb{V}\text{ar}\left[\widehat{z}\right] \approx \frac{\mathbb{V}\text{ar}\left[\widehat{\rho}\right]}{\rho^4}.


Estimator for the Evidence and Bayes Factor
===========================================


For independent, uncorrelated random variables one can compute the second 
order taylor expansion of the expectation and variance such that

.. math::
	:nowrap:

	\begin{align}
	\mathbb{E}\left[\frac{X}{Y}\right] &\simeq \frac{\mathbb{E}[X]}{\mathbb{E}[Y]} + \frac{\mathbb{E}[X]}{\mathbb{E}[Y]^3} \sigma_Y^2, \\
	\mathbb{V}\text{ar}\left(\frac{X}{Y}\right) &\simeq \frac{1}{\mathbb{E}[Y]^2} \sigma_X^2 + \frac{\mathbb{E}[X]^2}{\mathbb{E}[Y]^4} \sigma_Y^2.
	\end{align}

For a single evidence (*i.e.* :math:`Y = \rho`):
	
.. math::
	:nowrap:

	\begin{align}
	\mathbb{E}\left(\frac{1}{Y}\right) &\simeq \frac{1}{\mathbb{E}[Y]} \left( 1 + \frac{\sigma_Y^2}{\mathbb{E}[Y]^2} \right) \\
	\mathbb{V}\text{ar}\left(\frac{1}{Y}\right) &\simeq \frac{\sigma_Y^2}{\mathbb{E}[Y]^4}
	\end{align}

For Bayes factor (*i.e.* :math:`Y = \rho_1, X = \rho_2`):
	
.. math::
	:nowrap:

	\begin{align}
	\mathbb{E}\left(\frac{Z_1}{Z_2}\right) &= \mathbb{E}\left(\frac{\rho_2}{\rho_1}\right) = \mathbb{E}\left(\frac{X}{Y}\right) \simeq \frac{\mathbb{E}[X]}{\mathbb{E}[Y]} \left( 1 + \frac{\sigma_Y^2}{\mathbb{E}[Y]^2}  \right) \\
	\mathbb{V}\text{ar}\left(\frac{Z_1}{Z_2}\right) &= \mathbb{V}\text{ar}\left(\frac{\rho_2}{\rho_1}\right) = \mathbb{V}\text{ar}\left(\frac{X}{Y}\right) \simeq \frac{1}{\mathbb{E}[Y]^2} \sigma_X^2 + \frac{\mathbb{E}[X]^2}{\mathbb{E}[Y]^4} \sigma_Y^2 = \frac{\mathbb{E}[Y]^2 \sigma_X^2+ \mathbb{E}[X]^2 \sigma_Y^2}{\mathbb{E}[Y]^4}
	\end{align}

