**************************
Jupyter notebooks
**************************
These python notebooks can be found in the main code repository, under *notebooks*, and are fully interactable. We also provide Google collab extensions, which can be found in the introduction to each tutorial. They provide a high-level introduction to interfacing with **harmonic**, ranging from how to easily pick up **harmonic** and apply it to a project to more involved checkpointing and cross-validation for large scale, complex inferences.

One should note that **harmonic** acts on chains of posterior samples to infer the evidence, and is thus agnostic to the select sampler. In these notebooks we adopt the popular and highly accessible `emcee  <http://dfm.io/emcee/current/>`_ package for sampling, though alternate (perhaps custom) samplers are equally applicable.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   basic_usage.nblink
   cross-validation_hyper-parameters.nblink
   cross-validation_learnt_model.nblink
   checkpointing.nblink
   