|GitHub| |Build Status| |Docs| |CodeCov| |PyPI| |GPL license| |ArXiv|

.. |GitHub| image:: https://img.shields.io/badge/GitHub-harmonic-brightgreen.svg?style=flat
   :target: https://github.com/astro-informatics/harmonic

.. |Build Status| image:: https://github.com/astro-informatics/harmonic/actions/workflows/python.yml/badge.svg
   :target: https://github.com/astro-informatics/harmonic/actions/workflows/python.yml

.. |CodeCov| image:: https://codecov.io/gh/astro-informatics/harmonic/branch/main/graph/badge.svg?token=1s4SATphHV
   :target: https://codecov.io/gh/astro-informatics/harmonic

.. |ArXiv| image:: http://img.shields.io/badge/arXiv-20XX.XXXXX-orange.svg?style=flat
   :target: https://arxiv.org/abs/20XX.XXXXX

.. |GPL license| image:: https://img.shields.io/badge/License-GPL-blue.svg
   :target: http://perso.crans.org/besson/LICENSE.html

.. |Docs| image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
   :target: https://astro-informatics.github.io/harmonic/

.. |PyPI| image:: https://badge.fury.io/py/harmonic.svg
   :target: https://badge.fury.io/py/harmonic

Harmonic
========

We resurrect the infamous harmonic mean estimator for computing the marginal likelihood (Bayesian evidence) and solve its problematic large variance.  The marginal likelihood is a key component of Bayesian model selection since it is required to evaluate model posterior probabilities;  however, its computation is challenging.  The original harmonic mean estimator, first proposed in 1994 by Newton and Raftery, involves computing the harmonic mean of the likelihood given samples from the posterior.  It was immediately realised that the original estimator can fail catastrophically since its variance can become very large and may not be finite.  A number of variants of the harmonic mean estimator have been proposed to address this issue although none have proven fully satisfactory. 

We present the *learnt harmonic mean estimator*, a variant of the original estimator that solves its large variance problem.  This is achieved by interpreting the harmonic mean estimator as importance sampling and introducing a new target distribution.  The new target distribution is learned to approximate the optimal but inaccessible target, while minimising the variance of the resulting estimator.  Since the estimator requires samples of the posterior only it is agnostic to the strategy used to generate posterior samples. 

How to use this guide
---------------------
To get started, follow the :ref:`installation guide <Installation>`.  For a brief background of the *learnt harmonic mean estimator*  please see the :ref:`background <Background>` section of this guide, which provides sufficient background to inform the practioner.  For further background details please see the related paper (`McEwen et al. 2021 <TBC>`_).  We have also provided :ref:`tutorials <Jupyter Notebooks>`, comprised of a number of interactive notebooks that provide a step-by-step guide to get **harmonic** up and running for your particular application.  An up-to-date catalog of the software functionality can be found on the :ref:`API <Namespaces>` page. 

Basic usage
-----------
First you will want to install **harmonic**, which is as simple as running:

.. code-block:: bash
   
   pip install harmonic

Now, suppose you have collected many posterior samples, perhaps drawn using `emcee <https://emcee.readthedocs.io/en/stable/>`_ or another sampler, and *a posteriori* wish to compute the evidence, you would do something like:

.. code-block:: python
      
   import numpy as np 
   import harmonic as hm 

   # Instantiate harmonic's chains class
   chains = hm.Chains(n)
   chains.add_chains_3d(samples, lnprob)

   # Split the chains into the ones which will be used to train the machine learning model and for inference
   chains_train, chains_infer = hm.utils.split_data(chains, training_proportion=0.5)

   # Select a machine learning model and train it 
   model = hm.model.[select a model](model hyper-parameters)
   fit_success = model.fit(chains_train.samples, chains_train.ln_posterior)

   # Instantiate harmonic's evidence class
   ev = hm.Evidence(chains_infer.nchains, model)

   # Pass the evidence class the inference chains and compute the evidence!
   ev.add_chains(chains_infer)
   evidence, evidence_std = ev.compute_evidence()

.. note:: A variety of bespoke machine learning models are already supported, and the user is free to overload this functionality with their own models.

Referencing
---------------------

.. bibliography:: 
    :notcited:
    :list: bullet

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Background

   background/Harmonic_Estimator/original_harmonic_mean
   background/Harmonic_Estimator/retargetd_harmonic_mean
   background/Harmonic_Estimator/learnt_harmonic_mean
   background/Machine_Learning/index
   background/Bayes_Factors/bayes_factors


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials
   
   tutorials/basic_usage.nblink
   tutorials/cross-validation_hyper-parameters.nblink
   tutorials/cross-validation_learnt_model.nblink
   tutorials/checkpointing.nblink


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   api/index



