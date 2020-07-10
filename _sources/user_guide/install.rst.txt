.. _install:

Installation
============


.. _dependencies:

Installing dependencies
-----------------------

.. code-block:: bash

    pip install -r requirements.txt

Dependencies for examples
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -r requirements-examples.txt

Dependencies required for coverage testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -r requirements-extra.txt


.. _source:

Building from source
--------------------

.. code-block:: bash

    git clone https://github.com/astro-informatics/harmonic
    cd harmonic
    python setup.py build_ext --inplace


Code coverage
^^^^^^^^^^^^^

.. code-block:: bash

    python setup.py build_ext --inplace --define CYTHON_TRACE



Testing
-------

.. code-block:: bash

    pytest

Code coverage
^^^^^^^^^^^^^

.. code-block:: bash

   pytest --cov-report term --cov=harmonic --cov-config=.coveragerc


