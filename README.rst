.. image:: https://img.shields.io/badge/GitHub-harmonic-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/harmonic
.. image:: https://github.com/astro-informatics/harmonic/actions/workflows/python.yml/badge.svg
    :target: https://github.com/astro-informatics/harmonic/actions/workflows/python.yml
.. image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
    :target: https://astro-informatics.github.io/harmonic/
.. image:: https://codecov.io/gh/astro-informatics/harmonic/branch/main/graph/badge.svg?token=1s4SATphHV
    :target: https://codecov.io/gh/astro-informatics/harmonic
.. image:: https://badge.fury.io/py/harmonic.svg
    :target: https://badge.fury.io/py/harmonic
.. image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. image:: http://img.shields.io/badge/arXiv-2111.12720-orange.svg?style=flat
    :target: https://arxiv.org/abs/2111.12720
.. image:: http://img.shields.io/badge/arXiv-2207.04037-red.svg?style=flat
    :target: https://arxiv.org/abs/2207.04037
.. .. image:: https://img.shields.io/pypi/pyversions/harmonic.svg
..     :target: https://pypi.python.org/pypi/harmonic/

|logo| Harmonic
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/harm_badge_simple.svg" align="center" height="52" width="52">

``harmonic`` is an open source, well tested and documented Python implementation of the *learnt harmonic mean estimator* (`McEwen et al. 2021 <https://arxiv.org/abs/2111.12720>`_) to compute the marginal likelihood (Bayesian evidence), required for Bayesian model selection.

For an accessible overview of the *learnt harmonic mean estimator* please see this `Towards Data Science article <https://towardsdatascience.com/learnt-harmonic-mean-estimator-for-bayesian-model-selection-47258bb0fc2e>`_.

While ``harmonic`` requires only posterior samples, and so is agnostic to the technique used to perform Markov chain Monte Carlo (MCMC) sampling, ``harmonic`` works exceptionally well with MCMC sampling techniques that naturally provide samples from multiple chains by their ensemble nature, such as affine invariant ensemble samplers.  We therefore advocate use of `harmonic` with the popular `emcee <https://github.com/dfm/emcee>`_ code implementing the affine invariant sampler of `Goodman & Weare (2010) <https://cims.nyu.edu/~weare/papers/d13.pdf>`_.

Basic usage is highlighted in this `interactive demo <https://colab.research.google.com/github/astro-informatics/harmonic/blob/main/notebooks/basic_usage.ipynb>`_. 

Overview video
==============

.. image:: docs/assets/video_screenshot.png
    :target: https://www.youtube.com/watch?v=RHoQItSA4J4


Installation
============

Brief installation instructions are given below (for further details see the `full installation documentation <https://astro-informatics.github.io/harmonic/user_guide/install.html>`_).  

Quick install (PyPi)
--------------------
The ``harmonic`` package can be installed by running

.. code-block:: bash
    
    pip install harmonic

Install from source (GitHub)
----------------------------
The ``harmonic`` package can also be installed from source by running

.. code-block:: bash

    git clone https://github.com/astro-informatics/harmonic
    cd harmonic

and running the install script, within the root directory, with one command 

.. code-block:: bash

    bash build_harmonic.sh

To check the install has worked correctly run the unit tests with 

.. code-block:: bash

    pytest 
    
Documentation
=============

Comprehensive  `documentation for harmonic <https://astro-informatics.github.io/harmonic/>`_ is available.

Contributors
============

`Jason D. McEwen <http://www.jasonmcewen.org/>`_, `Christopher G. R. Wallis <https://scholar.google.co.uk/citations?user=Igl7nakAAAAJ&hl=en>`_, `Matthew A. Price <https://cosmomatt.github.io/>`_, `Matthew M. Docherty <https://mdochertyastro.com/>`_, `Alessio Spurio Mancini <https://www.alessiospuriomancini.com/>`_


Attribution
===========

Please cite `McEwen et al. (2021) <https://arxiv.org/abs/2111.12720>`_ if this code package has been of use in your project. 

A BibTeX entry for the paper is:

.. code-block:: 

     @article{harmonic, 
        author = {Jason~D.~McEwen and Christopher~G.~R.~Wallis and Matthew~A.~Price and Matthew~M.~Docherty},
         title = {Machine learning assisted {B}ayesian model comparison: learnt harmonic mean estimator},
       journal = {ArXiv},
        eprint = {arXiv:2111.12720},
          year = 2021
     }

Please *also* cite `Spurio Mancini et al. (2022) <https://arxiv.org/abs/2207.04037>`_ if this code has been of use in a simulation-based inference project.

A BibTeX entry for the paper is:

.. code-block::

     @article{harmonic_sbi,
        author = {Spurio Mancini, A. and Docherty, M. M. and Price, M. A. and McEwen, J. D.},
         title = {{B}ayesian model comparison for simulation-based inference},
       journal = {ArXiv},
        eprint = {arXiv:2207.04037},
          year = 2022
     }

License
=======

``harmonic`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/harmonic/blob/main/LICENSE.txt>`_), subject to 
the non-commercial use condition (see `LICENSE_EXT.txt <https://github.com/astro-informatics/harmonic/blob/main/LICENSE_EXT.txt>`_)

.. code-block::

     harmonic
     Copyright (C) 2021 Jason D. McEwen, Christopher G. R. Wallis, 
     Matthew A. Price, Matthew M. Docherty, Alessio Spurio Mancini & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
