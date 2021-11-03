|logo| Harmonic
=================================================================================================================

.. image:: ./docs/assets/harm_badge_simple.svg
    :width: 52%
    :height: 52%
    :align: center

.. image:: https://img.shields.io/badge/GitHub-harmonic-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/harmonic
.. image:: https://github.com/astro-informatics/harmonic/actions/workflows/python.yml/badge.svg
    :target: https://github.com/astro-informatics/harmonic/actions/workflows/python.yml
.. image:: https://codecov.io/gh/astro-informatics/harmonic/branch/master/graph/badge.svg?token=1s4SATphHV
    :target: https://codecov.io/gh/astro-informatics/harmonic
.. image:: http://img.shields.io/badge/arXiv-20XX.XXXXX-orange.svg?style=flat
    :target: https://arxiv.org/abs/20XX.XXXXX
.. image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html

Harmonic is an open source and fully documented python implementation of the Learnt Harmonic Mean estimator for the 
Bayesian evidence or marginal likelihood. In practice one uses chains gathered separately through MCMC sampling software 
to train one of the Harmonic machine learning models which then stabilize the harmonic mean estimator.

Documentation
-------------

See comprehensive documentation at `Harmonic Documentation <https://astro-informatics.github.io/harmonic/>`_.

Contributors
------------

`Jason McEwen <http://www.jasonmcewen.org/>`_, `Christopher Wallis <https://scholar.google.co.uk/citations?user=Igl7nakAAAAJ&hl=en>`_, `Matthew Price <https://scholar.google.co.uk/citations?user=w7_VDLQAAAAJ&hl=en&authuser=1>`_, `Matthew Docherty <https://mdochertyastro.com/>`_

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

Harmonic is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/harmonic/blob/master/LICENSE.txt>`_), subject to 
the non-commercial use condition (see `LICENSE_EXT.txt <https://github.com/astro-informatics/harmonic/blob/master/LICENSE_EXT.txt>`_)

.. code-block::

     harmonic
     Copyright (C) 2021 Jason McEwen & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
