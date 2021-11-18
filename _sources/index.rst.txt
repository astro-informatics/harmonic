|GitHub| |Build Status| |CodeCov| |ArXiv| |GPL license| |Docs| |PyPI|

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

The harmonic mean is an infamous estimator for the Bayesian evidence, first proposed by `Newton and Raftery (1994)  <https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j.2517-6161.1994.tb01956.x>`_. Whilst the harmonic mean estimator is asymptotically consistent it can fail catastrophically, depending on how one configures the estimator. By integrating bespoke machine learning techniques, **Harmonic** overcomes this hurdle by learning from disperate samples of the posterior. Our learnt estimator is agnostic to the sampler, provides accurate estimates of the variance and variance in the variance, and can scale into thousands of dimensions.

How to Use This Guide
---------------------
To get started you will first need to follow the :ref:`installation guide <Installation>`, following which it is recommended you run the testing suite to ensure your installation has been successful. Next, we have provided a :ref:`mini-tutorial <Jupyter Notebooks>`, comprised of 4 interactive notebooks, which will provide a step-by-step guide to get Harmonic up and running for your particular application. Finally, to see how Harmonic can be applied to various popular benchmark problems one should look to the :ref:`benchmark examples <Benchmark Examples>` page. An up-to-date catalog of the software functionality can be found on the :ref:`API <Namespaces>` page. 

Basic Usage
-----------
Suppose you have collected many samples, perhaps through `emcee <https://emcee.readthedocs.io/en/stable/>`_ or another sampler, from some generic :math:`n`-dimensional posterior and *a posteriori* wish to compute the evidence which we will assume is non-analytic, you would do something like:

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
   :caption: Mathematics

   background/Harmonic_Estimator/index
   background/Machine_Learning/index


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials
   
   tutorials/index

   
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   api/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Changelog

   api/changes



