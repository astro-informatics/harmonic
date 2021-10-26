**************************
Pathological Examples
**************************
To demonstrate the efficacy of **Harmonic** we have included a variety of common evidence estimation examples, where the posterior function is particularly pathological. These examples are somewhat standard benchmarks, and in many cases have historically highlight the failings of the vanilla harmonic mean estimator. Complexity increases from left to right, culminating in the logistic regression and non-nested linear regressions of the Pima Indian and Radiata Pine benchmarks respectively. See `Friel and Wyse (2011)  <https://arxiv.org/pdf/1111.1957.pdf>`_ for an extensive review of various estimators applied to these benchmarks.

.. automodule:: harmonic

.. tabs:: 
	
	.. tab:: Gaussian

		.. include:: Gaussian/Example_code/index.rst

	.. tab:: Rosenbrock

		.. include:: Rosenbrock/Example_code/index.rst

	.. tab:: Rastrigin

		.. include:: Rastrigin/Example_code/index.rst

	.. tab:: Normal Gamma

		.. include:: Normal_gamma/Example_code/index.rst

	.. tab:: Radiata Pine

		.. include:: Radiata_pine/Example_code/index.rst

	.. tab:: Pima Indian

		.. include:: Pima_indian/Example_code/index.rst

..
	.. toctree::
	   :maxdepth: 1
	   :caption: Contents:

	   Gaussian/index
	   Rosenbrock/index
	   Rastrigin/index
	   Normal_gamma/index
	   Radiata_pine/index
	   Pima_indian/index