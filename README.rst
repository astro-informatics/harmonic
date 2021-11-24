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
.. image:: http://img.shields.io/badge/arXiv-20XX.XXXXX-orange.svg?style=flat
    :target: https://arxiv.org/abs/20XX.XXXXX
.. .. image:: https://img.shields.io/pypi/pyversions/harmonic.svg
..     :target: https://pypi.python.org/pypi/harmonic/

|logo| Harmonic
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/harm_badge_simple.svg" align="center" height="52" width="52">

``harmonic`` is an open source, well tested and documented Python implementation of the *learnt harmonic mean estimator* (`McEwen et al. 2021 <TBC>`_) to compute the marginal likelihood (Bayesian evidence), required for Bayesian model selection.

While ``harmonic`` requires only posterior samples, and so is agnostic to the technique used to perform Markov chain Monte Carlo (MCMC) sampling, ``harmonic`` works exceptionally well with MCMC sampling techniques that naturally provide samples from multiple chains by their ensemble nature, such as affine invariant ensemble samplers.  We therefore advocate use of `harmonic` with the popular `emcee <https://github.com/dfm/emcee>`_ code implementing the affine invariant sampler of `Goodman & Weare (2010) <https://cims.nyu.edu/~weare/papers/d13.pdf>`_.

Basic usage is highlighted in this `interactive demo <https://colab.research.google.com/github/astro-informatics/harmonic/blob/main/notebooks/basic_usage.ipynb>`_. 

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

`Jason D. McEwen <http://www.jasonmcewen.org/>`_, `Christopher G. R. Wallis <https://scholar.google.co.uk/citations?user=Igl7nakAAAAJ&hl=en>`_, `Matthew A. Price <https://scholar.google.co.uk/citations?user=w7_VDLQAAAAJ&hl=en&authuser=1>`_, `Matthew M. Docherty <https://mdochertyastro.com/>`_

Attribution
===========

Please cite `McEwen et al. (2021) <TBC>`_ if this code package has been of use in your project. 

A BibTeX entry for the paper is:

.. code-block:: 

     @article{harmonic, 
        author = {Jason.D.~McEwen and Christopher~G.~R.~Wallis and Matthew~A.~Price and Matthew~M.~Docherty},
         title = {Machine learning assisted Bayesian model comparison: learnt harmonic mean estimator},
       journal = {ArXiv},
        eprint = {arXiv:XXXX.XXXX},
          year = 2021
     }

License
=======

``harmonic`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/harmonic/blob/main/LICENSE.txt>`_), subject to 
the non-commercial use condition (see `LICENSE_EXT.txt <https://github.com/astro-informatics/harmonic/blob/main/LICENSE_EXT.txt>`_)

.. code-block::

     harmonic
     Copyright (C) 2021 Jason D. McEwen, Christopher G. R. Wallis, 
     Matthew A. Price, Matthew M. Docherty & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
