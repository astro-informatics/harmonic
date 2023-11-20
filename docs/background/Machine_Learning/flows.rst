
Normalizing flows are a class of probabilistic models that allow one to evaluate the density of and sample from a learned probability distribution (for a review see `(Papamakarios et al., 2021) <https://arxiv.org/abs/1912.02762>`_). They consist of a series of transformations that are applied to a simple base distribution.  A vector :math:`\theta` 
of an unknown distribution :math:`p(\theta)`, can be expressed through a transformation :math:`T` of a vector :math:`z` sampled from a base distribution :math:`q(z)`:

.. math::
	\theta = T(z), \text{ where } z \sim q(z).

Typically the base distribution is chosen so that its density can be evaluated simply and that it can be sampled from easily. Often a Gaussian is used for the base distribution.
The unknown distribution can then be recovered by the change of variables formula:

.. math::
	p(\theta) = q(z) \vert \det J_{T}(z)  \vert^{-1},

where :math:`J_{T}(z)` is the Jacobian corresponding to transformation :math:`T`. In a flow-based model :math:`T` consists of a series of learned transformations that are each invertible and differentiable, so that the full transformation is also invertible and differentiable.  This allows us to compose multiple simple transformations with learned parameters, into what is called a flow, obtaining a normalized approximation of the unknown distribution that we can sample from and evaluate.  Careful attention is given to construction of the transformations such that the determinant of the Jacobian can be computed easily.

**Real NVP Flows**

A relatively simple example of a normalizing flow is the real-valued non-volume preserving (real NVP) flow introduced in `(Dinh et al., 2016) <https://arxiv.org/abs/1605.08803>`_.
It consists of a series of bijective transformations given by affine coupling layers. Consider the :math:`D` dimensional input :math:`z`, split into elements up to and following :math:`d`, respectively, :math:`z_{1:d}` and :math:`z_{d+1:D}`, for :math:`d<D`.  Given input :math:`z`, the output :math:`y` of an affine couple layer is calculated by

.. math::
	y_{1:d} =   & z_{1:d} ;                                                 \\
	y_{d+1:D} = & z_{d+1:D} \odot \exp\bigl(s(z_{1:d})\bigr) +  t(z_{1:d}),

where :math:`\odot` denotes Hadamard (elementwise) multiplication.
The scale :math:`s` and translation :math:`t` are typically represented by neural networks with learnable parameters that take as input :math:`z_{1:d}`.  This construction is easily invertible and ensures the Jacobian is a lower-triangular matrix, making its determinant efficient to calculate.


**Rational Quadratic Spline Flows**

A more complex and expressive class of flows are rational quadratic spline flows described in detail in `(Durkan et al., 2019) <https://arxiv.org/abs/1906.04032>`_. The architecture is similar to Real NVP flows, but the layers include monotonic splines. These are piecewise functions consisting of multiple segments of monotonic rational-quadratics with learned parameters. Such layers are combined with alternating affine transformations to create the normalizing flow. Rational quadratic spline flows are well-suited to higher dimensional and more complex problems than real NVP flows.