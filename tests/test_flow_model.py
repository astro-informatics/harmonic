import pytest
import harmonic.model_nf as model_nf
import jax.numpy as jnp
import jax
import harmonic as hm


# TODO: move this to conftest.py to follow best practices.
def standard_nd_gaussian_pdf(x):
    """
    Calculate the probability density function (PDF) of an n-dimensional Gaussian
    distribution with zero mean and unit covariance.

    Parameters:
    - x: Input vector of length n.

    Returns:
    - pdf: log PDF value at input vector x.
    """
    n = len(x)

    # The normalizing constant (coefficient)
    C = -jnp.log(2 * jnp.pi) * n / 2

    # Calculate the Mahalanobis distance
    mahalanobis_dist = jnp.dot(x, x)

    # Calculate the PDF value
    pdf = C - 0.5 * mahalanobis_dist

    return pdf


def test_FlowModel_constructor():
    with pytest.raises(NotImplementedError):
        ndim = 3
        model = model_nf.FlowModel(ndim)
        training_samples = jnp.zeros((12, ndim))
        model.fit(training_samples)


def test_RealNVP_constructor():
    with pytest.raises(ValueError):
        RealNVP = model_nf.RealNVPModel(0)

    with pytest.raises(ValueError):
        RealNVP = model_nf.RealNVPModel(-1)

    ndim = 3

    with pytest.raises(ValueError):
        RealNVP = model_nf.RealNVPModel(ndim, n_scaled_layers=0)

    RealNVP = model_nf.RealNVPModel(ndim, standardize=True)

    with pytest.raises(ValueError):
        training_samples = jnp.zeros((12, ndim + 1))
        RealNVP.fit(training_samples)

    with pytest.raises(ValueError):
        RealNVP.temperature = 1.2
        RealNVP.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        RealNVP.temperature = -0.5
        RealNVP.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        RealNVP.temperature = 0
        RealNVP.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        RealNVP.temperature = 1.2
        RealNVP.sample(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        RealNVP.temperature = -0.5
        RealNVP.sample(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        RealNVP.temperature = 0
        RealNVP.sample(jnp.zeros(ndim))

    assert RealNVP.is_fitted() == False
    training_samples = jnp.zeros((12, ndim))
    RealNVP.fit(training_samples, verbose=True, epochs=5)
    assert RealNVP.is_fitted() == True


def test_RealNVP_flow():
    with pytest.raises(ValueError):
        flow = hm.flows.RealNVP(3, n_scaled_layers=0)
        flow.make_flow()


def test_RQSpline_constructor():
    with pytest.raises(ValueError):
        spline = model_nf.RQSplineModel(0)

    with pytest.raises(ValueError):
        spline = model_nf.RQSplineModel(-1)

    ndim = 3
    spline = model_nf.RQSplineModel(ndim, standardize=True)

    with pytest.raises(ValueError):
        training_samples = jnp.zeros((12, ndim + 1))
        spline.fit(training_samples)

    with pytest.raises(ValueError):
        spline.temperature = 1.2
        spline.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        spline.temperature = -0.5
        spline.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        spline.temperature = 0
        spline.predict(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        spline.temperature = 1.2
        spline.sample(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        spline.temperature = -0.5
        spline.sample(jnp.zeros(ndim))

    with pytest.raises(ValueError):
        spline.temperature = 0
        spline.sample(jnp.zeros(ndim))

    assert spline.is_fitted() == False
    training_samples = jnp.zeros((12, ndim))
    spline.fit(training_samples)
    assert spline.is_fitted() == True


# TODO: combine tests into one test with a model variable.
def test_RealNVP_gaussian():
    # Define the number of dimensions and the mean of the Gaussian
    ndim = 2
    num_samples = 10000
    epochs = 80
    # Initialize a PRNG key (you can use any valid key)
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(ndim)
    cov = jnp.eye(ndim)

    # Generate random samples from the 2D Gaussian distribution
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,))

    RealNVP = model_nf.RealNVPModel(ndim, standardize=True)
    RealNVP.fit(samples, epochs=epochs, verbose=True)

    nsamples = 5000
    RealNVP.temperature = 1.0
    flow_samples = RealNVP.sample(nsamples)
    sample_var = jnp.var(flow_samples, axis=0)
    sample_mean = jnp.mean(flow_samples, axis=0)

    test = jnp.ones(ndim) * 0.2
    assert jnp.exp(RealNVP.predict(jnp.array([test])))[0] == pytest.approx(
        jnp.exp(standard_nd_gaussian_pdf(test)), rel=0.1
    ), "Real NVP probability density not in agreement with analytical value"

    for i in range(ndim):
        assert sample_mean[i] == pytest.approx(0.0, abs=0.11), (
            "Sample mean in dimension " + str(i) + " is " + str(sample_mean[i])
        )
        assert sample_var[i] == pytest.approx(1.0, abs=0.11), (
            "Sample variance in dimension " + str(i) + " is " + str(sample_var[i])
        )

    RealNVP.temperature = 0.8
    flow_samples_concentrated = RealNVP.sample(nsamples)
    sample_var_concentrated = jnp.var(flow_samples_concentrated, axis=0)

    for i in range(ndim):
        assert (
            sample_var[i] > sample_var_concentrated[i]
        ), "Reducing temperature increases variance in dimension " + str(i)


def test_RQSpline_gaussian():
    # Define the number of dimensions and the mean of the Gaussian
    ndim = 2
    num_samples = 10000
    epochs = 20
    # Initialize a PRNG key (you can use any valid key)
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(ndim)
    cov = jnp.eye(ndim)

    # Generate random samples from the 2D Gaussian distribution
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,))

    spline = model_nf.RQSplineModel(ndim, standardize=True)
    spline.fit(samples, epochs=epochs, verbose=True)

    nsamples = 5000
    spline.temperature = 1.0
    flow_samples = spline.sample(nsamples)
    sample_var = jnp.var(flow_samples, axis=0)
    sample_mean = jnp.mean(flow_samples, axis=0)

    for i in range(ndim):
        assert sample_mean[i] == pytest.approx(0.0, abs=0.11), (
            "Sample mean in dimension " + str(i) + " is " + str(sample_mean[i])
        )
        assert sample_var[i] == pytest.approx(1.0, abs=0.11), (
            "Sample variance in dimension " + str(i) + " is " + str(sample_var[i])
        )

    test = jnp.ones(ndim) * 0.2
    assert jnp.exp(spline.predict(jnp.array([test])))[0] == pytest.approx(
        jnp.exp(standard_nd_gaussian_pdf(test)), rel=0.1
    ), "Spline probability density not in agreement with analytical value"

    spline.temperature = 0.8
    flow_samples_concentrated = spline.sample(nsamples)
    sample_var_concentrated = jnp.var(flow_samples_concentrated, axis=0)

    for i in range(ndim):
        assert (
            sample_var[i] > sample_var_concentrated[i]
        ), "Reducing temperature increases variance in dimension " + str(i)


def test_model_serialization():
    # Define the number of dimensions and the mean of the Gaussian
    ndim = 2
    num_samples = 100
    epochs_NVP = 50
    epochs_spline = 5
    # Initialize a PRNG key (you can use any valid key)
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(ndim)
    cov = jnp.eye(ndim)

    # Generate random samples from the 2D Gaussian distribution
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,))

    # NVP params
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
    var_scale = 0.6

    model_NVP = model_nf.RealNVPModel(
        ndim,
        n_scaled_layers=n_scaled,
        n_unscaled_layers=n_unscaled,
        learning_rate=learning_rate,
        momentum=momentum,
        standardize=standardize,
        temperature=var_scale,
    )

    model_NVP.fit(samples, epochs=epochs_NVP)

    # Serialize model
    model_NVP.serialize(".test.dat")

    # Deserialize model
    model_NVP2 = model_nf.RealNVPModel.deserialize(".test.dat")

    assert model_NVP2.ndim == model_NVP.ndim
    assert model_NVP2.is_fitted() == model_NVP.is_fitted()
    assert model_NVP2.n_scaled_layers == model_NVP.n_scaled_layers
    assert model_NVP2.n_unscaled_layers == model_NVP.n_unscaled_layers
    assert model_NVP2.learning_rate == model_NVP.learning_rate
    assert model_NVP2.momentum == model_NVP.momentum
    assert model_NVP2.standardize == model_NVP.standardize
    assert model_NVP2.temperature == model_NVP.temperature

    test = jnp.array([jnp.ones(ndim)])
    assert model_NVP2.predict(test) == model_NVP.predict(test), (
        "Prediction for deserialized model is "
        + str(model_NVP2.predict(test))
        + ", not equal to "
        + str(model_NVP.predict(test))
    )

    model_spline = model_nf.RQSplineModel(
        ndim,
        n_layers=n_layers,
        n_bins=n_bins,
        hidden_size=hidden_size,
        spline_range=spline_range,
        standardize=standardize,
        learning_rate=learning_rate,
        momentum=momentum,
        temperature=var_scale,
    )
    model_spline.fit(samples, epochs=epochs_spline)
    # Serialize model
    model_spline.serialize(".test.dat")

    # Deserialize model
    model_spline2 = model_nf.RQSplineModel.deserialize(".test.dat")
    assert model_spline2.ndim == model_spline.ndim
    assert model_spline2.is_fitted() == model_spline.is_fitted()
    assert model_spline2.n_layers == model_spline.n_layers
    assert model_spline2.n_bins == model_spline.n_bins
    assert model_spline2.hidden_size == model_spline.hidden_size
    assert model_spline2.spline_range == model_spline.spline_range
    assert model_spline2.learning_rate == model_spline.learning_rate
    assert model_spline2.momentum == model_spline.momentum
    assert model_spline2.standardize == model_spline.standardize
    assert model_spline2.temperature == model_spline.temperature

    assert model_spline2.predict(test) == model_spline.predict(test)
