import pytest
import harmonic.model as md
import jax.numpy as jnp
import jax
import harmonic as hm

real_nvp_2D = md.RealNVPModel(2, standardize=True)
spline_4D = md.RQSplineModel(4, n_layers=2, n_bins=64, standardize=True)
spline_3D  = md.RQSplineModel(3,n_layers=2, n_bins=64, standardize=False)

model_classes = [md.RealNVPModel, md.RQSplineModel]

models_to_test = [real_nvp_2D, spline_4D]
models_to_test1 = [spline_4D, spline_3D]
gaussian_var = [0.1,0.5, 1.,10., 20.]

# Make models for serialization tests
# NVP params
ndim = 2
n_scaled = 13
n_unscaled = 6

# Spline params
n_layers = 13
n_bins = 8
hidden_size = [64, 64]
spline_range = (-10.0, 10.0)

# Optimizer params
learning_rate = 0.01
momentum = 0.8
standardize = True
temperature = 0.6

real_NVP_serialization = md.RealNVPModel(
    ndim,
    n_scaled_layers=n_scaled,
    n_unscaled_layers=n_unscaled,
    learning_rate=learning_rate,
    momentum=momentum,
    standardize=standardize,
    temperature=temperature,
)

spline_serialization = md.RQSplineModel(
    ndim,
    n_layers=n_layers,
    n_bins=n_bins,
    hidden_size=hidden_size,
    spline_range=spline_range,
    standardize=standardize,
    learning_rate=learning_rate,
    momentum=momentum,
    temperature=temperature,
)

models_serialization = [real_NVP_serialization, spline_serialization]


def standard_nd_gaussian_pdf(x, var=1.):
    """
    Calculate the probability density function (PDF) of an n-dimensional Gaussian
    distribution with zero mean and diagonal covariance with entries var.

    Parameters:
    - x: Input vector of length n.
    - var: Gaussian variance

    Returns:
    - pdf: log PDF value at input vector x.
    """
    n = len(x)

    # The normalizing constant (coefficient)
    C = -jnp.log(2 * jnp.pi* var) * n / 2

    # Calculate the Mahalanobis distance
    mahalanobis_dist = jnp.dot(x, x)/var

    # Calculate the PDF value
    pdf = C - 0.5 * mahalanobis_dist

    return pdf


def test_FlowModel_constructor():
    with pytest.raises(NotImplementedError):
        ndim = 3
        model = md.FlowModel(ndim)
        training_samples = jnp.zeros((12, ndim))
        model.fit(training_samples)


@pytest.mark.parametrize("model_class", model_classes)
def test_flow_constructor(model_class):
    with pytest.raises(ValueError):
        model = model_class(0)

    with pytest.raises(ValueError):
        model = model_class(-1)


def test_RealNVP_no_scaled_layers():
    ndim = 3
    with pytest.raises(ValueError):
        RealNVP = md.RealNVPModel(ndim, n_scaled_layers=0)

    with pytest.raises(ValueError):
        flow = hm.flows.RealNVP(ndim, n_scaled_layers=0)
        flow.make_flow()


@pytest.mark.parametrize("model", models_to_test)
def test_flow_errors(model):
    ndim = model.ndim

    with pytest.raises(ValueError):
        training_samples = jnp.zeros((12, ndim + 1))
        model.fit(training_samples)

    with pytest.raises(ValueError):
        model.temperature = 1.2
        model.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        model.temperature = -0.5
        model.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        model.temperature = 0
        model.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        model.temperature = 1.2
        model.sample(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        model.temperature = -0.5
        model.sample(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        model.temperature = 0
        model.sample(jnp.zeros(ndim))


@pytest.mark.parametrize("model", models_to_test)
def test_flow_is_fitted(model):
    ndim = model.ndim
    assert model.is_fitted() == False
    training_samples = jnp.zeros((12, ndim))
    model.fit(training_samples, verbose=True, epochs=5)
    assert model.is_fitted() == True


@pytest.mark.parametrize("model", models_to_test1)
@pytest.mark.parametrize("var", gaussian_var)
def test_flows_gaussian_pdf(model, var):
    # Define the number of dimensions and the mean of the Gaussian
    ndim = model.ndim
    num_samples = 10000

    if isinstance(model, md.RealNVPModel):
        epochs = 100
    elif isinstance(model, md.RQSplineModel):
        epochs = 30

    # Initialize a PRNG key (you can use any valid key)
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(ndim)
    cov = jnp.eye(ndim)*var

    # Generate random samples from the Gaussian distribution
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,))

    model.fit(samples, epochs=epochs, verbose=True)
    model.temperature = 1.0

    test = jnp.ones(ndim) * 0.2
    predicted_pdf = model.predict(test)
    analytic_pdf = standard_nd_gaussian_pdf(test, var=var)
    print("T ", var, "Predicted log pdf ", predicted_pdf, " Analytic log pdf", analytic_pdf)
    assert jnp.exp(predicted_pdf) == pytest.approx(jnp.exp(analytic_pdf), rel=0.15), "Flow probability density not in agreement with analytical value"

    temp = 0.5
    model.temperature = temp
    predicted_pdf = model.predict(test)
    analytic_pdf = standard_nd_gaussian_pdf(test, var=var*temp)
    print("T ", var, "Predicted log pdf ", predicted_pdf, " Analytic log pdf", analytic_pdf)
    assert jnp.exp(predicted_pdf) == pytest.approx(jnp.exp(analytic_pdf), rel=0.15), "Reduced flow probability density not in agreement with analytical value"

@pytest.mark.parametrize("model", models_to_test)
def test_flows_gaussian(model):
    # Define the number of dimensions and the mean of the Gaussian
    ndim = model.ndim
    num_samples = 10000

    if isinstance(model, md.RealNVPModel):
        epochs = 80
    elif isinstance(model, md.RQSplineModel):
        epochs = 20

    # Initialize a PRNG key (you can use any valid key)
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(ndim)
    cov = jnp.eye(ndim)

    # Generate random samples from the 2D Gaussian distribution
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,))

    model.fit(samples, epochs=epochs, verbose=True)

    nsamples = 5000
    model.temperature = 1.0
    flow_samples = model.sample(nsamples)
    sample_var = jnp.var(flow_samples, axis=0)
    sample_mean = jnp.mean(flow_samples, axis=0)

    for i in range(ndim):
        assert sample_mean[i] == pytest.approx(0.0, abs=0.15), (
            "Sample mean in dimension " + str(i) + " is " + str(sample_mean[i])
        )
        assert sample_var[i] == pytest.approx(1.0, abs=0.15), (
            "Sample variance in dimension " + str(i) + " is " + str(sample_var[i])
        )

    model.temperature = 0.8
    flow_samples_concentrated = model.sample(nsamples)
    sample_var_concentrated = jnp.var(flow_samples_concentrated, axis=0)

    for i in range(ndim):
        assert (
            sample_var[i] > sample_var_concentrated[i]
        ), "Reducing temperature increases variance in dimension " + str(i)


@pytest.mark.parametrize("model", models_serialization)
def test_model_serialization(model):
    # Define the number of dimensions and the mean of the Gaussian
    ndim = model.ndim
    num_samples = 100

    if isinstance(model, md.RealNVPModel):
        epochs = 50
        model_class = md.RealNVPModel
    elif isinstance(model, md.RQSplineModel):
        epochs = 5
        model_class = md.RQSplineModel

    # Initialize a PRNG key (you can use any valid key)
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(ndim)
    cov = jnp.eye(ndim)

    # Generate random samples from the 2D Gaussian distribution
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,))

    model.fit(samples, epochs=epochs)

    # Serialize model
    model.serialize(".test.dat")

    # Deserialize model
    model2 = model_class.deserialize(".test.dat")

    assert model2.ndim == model.ndim
    assert model2.is_fitted() == model.is_fitted()
    assert model2.learning_rate == model.learning_rate
    assert model2.momentum == model.momentum
    assert model2.standardize == model.standardize
    assert model2.temperature == model.temperature

    if isinstance(model, md.RealNVPModel):
        assert model2.n_scaled_layers == model.n_scaled_layers
        assert model2.n_unscaled_layers == model.n_unscaled_layers
    elif isinstance(model, md.RQSplineModel):
        assert model2.n_layers == model.n_layers
        assert model2.n_bins == model.n_bins
        assert model2.hidden_size == model.hidden_size
        assert model2.spline_range == model.spline_range

    test = jnp.ones(ndim)
    assert model2.predict(test) == model.predict(test), (
        "Prediction for deserialized model is "
        + str(model2.predict(test))
        + ", not equal to "
        + str(model.predict(test))
    )
