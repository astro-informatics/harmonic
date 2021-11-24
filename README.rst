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

``harmonic`` is an open source and fully documented python implementation of the Learnt Harmonic Mean estimator for the 
Bayesian evidence or marginal likelihood. In practice one uses chains gathered separately through MCMC sampling software 
to train one of the Harmonic machine learning models which then stabilize the harmonic mean estimator. Basic usage is
highlighted in this `Interactive Demo <https://colab.research.google.com/github/astro-informatics/harmonic/blob/main/notebooks/basic_usage.ipynb>`_. 

Installation
-------------

Quick Install (PyPi)
^^^^^^^^^^^^^^^^^^^^
The harmonic package can be quickly be installed by running

.. code-block:: bash
    
    pip install harmonic

Install From Source (GitHub)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The harmonic package can also be installed from source by running

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
-------------

See comprehensive documentation at `Harmonic Documentation <https://astro-informatics.github.io/harmonic/>`_.

Contributors
------------

`Jason D. McEwen <http://www.jasonmcewen.org/>`_, `Christopher G. R. Wallis <https://scholar.google.co.uk/citations?user=Igl7nakAAAAJ&hl=en>`_, `Matthew A. Price <https://scholar.google.co.uk/citations?user=w7_VDLQAAAAJ&hl=en&authuser=1>`_, `Matthew M. Docherty <https://mdochertyastro.com/>`_

Attribution
-----------

Please cite McEwen et al 2021 if this code package has been of use in any project. A link will be provided 
shortly upon submission. A BibTeX entry for the paper is:

.. code-block:: 

     @article{harmonic, 
        author = {{McEwen}, J.~D. and {Wallis}, C.~G.~R. and {Price}, M.~A.},
         title = {Machine learning assisted marginal likelihood estimation: 
                 learnt harmonic mean estimator},
       journal = {Bayesian Analysis in prep},
          year = 2021
     }

License
-------

``harmonic`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/harmonic/blob/main/LICENSE.txt>`_), subject to 
the non-commercial use condition (see `LICENSE_EXT.txt <https://github.com/astro-informatics/harmonic/blob/main/LICENSE_EXT.txt>`_)

.. code-block::

     harmonic
     Copyright (C) 2021 Jason D. McEwen, Christopher G. R. Wallis, Matthew A. Price, Matthew M. Docherty & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
