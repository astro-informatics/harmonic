.. _install:

Installation
============
We recommend installing **harmonic** through `PyPi <https://pypi.org>`_, however in some cases one may wish to install **harmonic** directly from source, which is also relatively straightforward. 

Quick install (PyPi)
--------------------
Install **harmonic** from PyPi with a single command

.. code-block:: bash

    pip install harmonic 

Check that the package has installed by running 

.. code-block:: bash 

	pip list 

and locate harmonic.


Install from source (GitHub)
----------------------------

When installing from source we recommend working within an existing conda environment, or creating a fresh conda environment to avoid any dependency conflicts,

.. code-block:: bash

    conda create -n harmonic_env python=3.8.0
    conda activate harmonic_env

Once within a fresh environment **harmonic** may be installed by cloning the GitHub repository

.. code-block:: bash

    git clone https://github.com/astro-informatics/harmonic
    cd harmonic

and installing within the root directory, with one command 

.. code-block:: bash

    bash build_harmonic.sh

To check the install has worked correctly run the unit tests with 

.. code-block:: bash

	pytest 

.. note:: For installing from source a conda environment is required by the installation bash script, which is recommended, due to a pandoc dependency.
