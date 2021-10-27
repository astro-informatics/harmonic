.. _install:

Installation
============
We recommend installing Harmonic through `PyPi <https://pypi.org>`_, however in some cases one may wish to install Harmonic directly from source, which is also relatively straightforward. In either case we recommend creating a fresh conda environment to avoid any dependency conflicts 

.. code-block:: bash

    conda create -n "env_name" python==3.8.0
    source activate env_name

Once within a fresh environment Harmonic may be installed by the following.

.. tabs::
	
	.. tab:: PyPi

		.. code-block:: bash

		    pip install harmonic 

	.. tab:: GitHub
	
		.. code-block:: bash

		    git clone https://github.com/astro-informatics/harmonic
		    cd harmonic
		    python setup.py build_ext --inplace


		Alternatively, if one wishes to inspect code coverage they should run

		.. code-block:: bash

		    git clone https://github.com/astro-informatics/harmonic
		    cd harmonic
		    python setup.py build_ext --inplace --define CYTHON_TRACE

		To import Harmonic from outside of the install directory one should run

		.. code-block:: bash 

		    pip install -e .

		from the root directory.

Testing
-------

The Harmonic test suite may be executed by running

.. code-block:: bash

    pytest

or alternatively with code-coverage by running 

.. code-block:: bash

   pytest --cov-report term --cov=harmonic --cov-config=.coveragerc

Dependencies
------------

This project has a variety of dependencies which are managed *via* `PyPi <https://pypi.org>`_. Some dependencies are required for **Harmonic** core functionality, others are required only for the testing suites and documentation. To install project dependencies simply run 

.. code-block:: bash

    pip install -r requirements[x].txt

from the project root directory, where an empty [x] installs core requirementes, [-examples] installs example requirements, and [-extra] installs requirements for coverage tests. Below is a categorical list of each dependencies.

.. tabs::
	
	.. tab:: Harmonic Core

		* python (>=3.8.12)
		* `scikit-learn <https://pypi.org/project/scikit-learn/>`_ (>=0.22.2.post1)
		* `scipy <https://pypi.org/project/scipy/>`_ (>=1.4.1)
		* `colorlog <https://pypi.org/project/colorlog/>`_ (>=4.1.0)
		* `pyyaml <https://pypi.org/project/PyYAML/>`_ (>=3.12)

	.. tab:: Examples

		* `emcee <https://pypi.org/project/emcee/>`_ (>=3.1.1)
		* `matplotlib <https://pypi.org/project/matplotlib/>`_ (>=3.4.3)
		* `corner <https://pypi.org/project/corner/>`_ (>=2.2.1)
		* `getdist <https://pypi.org/project/GetDist/>`_ (>=1.3.2)

	.. tab:: Test Suite

		* `pytest-cov <https://pypi.org/project/pytest-cov/>`_ (>=3.0.0)
		* `codecov <https://pypi.org/project/codecov/>`_ (>=2.1.12)

	.. tab:: Notebooks

		* `ipython <https://pypi.org/project/ipython/>`_ (>=7.16.1)
		* `jupyter <https://pypi.org/project/jupyter/>`_ (>=1.0.0)

	.. tab:: Documentation

		* `sphinx <https://pypi.org/project/Sphinx/>`_ (>=4.2.0)
		* `nbsphinx-link <https://pypi.org/project/nbsphinx-link/>`_ (>=1.3.0)
		* `pandoc <https://pypi.org/project/pandoc/>`_ (>=1.1.0)
		* `sphinx-rtd-theme <https://pypi.org/project/sphinx-rtd-theme/>`_ (>=1.0.0)
		* `sphinx-toolbox <https://pypi.org/project/sphinx-toolbox/>`_ (>=2.15.0)
		* `sphinx-tabs <https://pypi.org/project/sphinx-tabs/>`_ (>=3.2.0)
		* `sphinx-rtd-dark-mode <https://pypi.org/project/sphinx-rtd-dark-mode/>`_ (>=1.2.4)
		* `sphinxcontrib-bibtex <https://pypi.org/project/sphinxcontrib-bibtex/>`_ (>=2.4.1)



