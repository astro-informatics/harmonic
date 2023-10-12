import pytest
import harmonic.model_nf as model_nf
import jax.numpy as jnp
import jax

def test_RealNVP_constructor():

    with pytest.raises(ValueError):
        RealNVP = model_nf.RealNVPModel(0)

    with pytest.raises(ValueError):
        RealNVP = model_nf.RealNVPModel(-1)

    ndim = 3
    RealNVP = model_nf.RealNVPModel(ndim, standardize=True) 

    with pytest.raises(ValueError):
        training_samples = jnp.zeros((12,ndim+1))
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
    training_samples = jnp.zeros((12,ndim))
    RealNVP.fit(training_samples)
    assert RealNVP.is_fitted() == True


def test_RQSpline_constructor():

    with pytest.raises(ValueError):
        spline = model_nf.RQSplineFlow(0)

    with pytest.raises(ValueError):
        spline = model_nf.RQSplineFlow(-1)

    ndim = 3
    spline = model_nf.RQSplineFlow(ndim, standardize=True) 

    with pytest.raises(ValueError):
        training_samples = jnp.zeros((12,ndim+1))
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
    training_samples = jnp.zeros((12,ndim))
    spline.fit(training_samples)
    assert spline.is_fitted() == True




def test_RealNVP_gaussian():

    # Define the number of dimensions and the mean of the Gaussian
    ndim = 2
    num_samples = 10000
    epochs = 50
    # Initialize a PRNG key (you can use any valid key)
    key = jax.random.PRNGKey(0)

    # Generate random samples from the 2D Gaussian distribution
    samples = jax.random.normal(key, shape=(num_samples, ndim))

    RealNVP = model_nf.RealNVPModel(ndim, standardize=True)
    RealNVP.fit(samples, epochs=epochs)

    nsamples = 10000
    RealNVP.temperature = 1.
    flow_samples = RealNVP.sample(nsamples)
    sample_var = jnp.var(flow_samples, axis = 0)
    RealNVP.temperature = 0.8
    flow_samples_concentrated = RealNVP.sample(nsamples)
    sample_var_concentrated = jnp.var(flow_samples_concentrated, axis = 0)

    for i in range(ndim):
        assert sample_var[i] > sample_var_concentrated[i], "Reducing temperature increases variance in dimension " + str(i)
    
    sample_mean = jnp.mean(flow_samples, axis=0)

    for i in range(ndim):
        assert sample_mean[i] == pytest.approx(0.0, abs = 0.1), "Sample mean in dimension " + str(i) + " is " + str(sample_mean[i])
        assert sample_var[i] == pytest.approx(1.0, abs = 0.1), "Sample variance in dimension " + str(i) + " is " + str(sample_var[i])



def test_RQSpline_gaussian():

    # Define the number of dimensions and the mean of the Gaussian
    ndim = 2
    num_samples = 10000
    epochs = 20
    # Initialize a PRNG key (you can use any valid key)
    key = jax.random.PRNGKey(0)

    # Generate random samples from the 2D Gaussian distribution
    samples = jax.random.normal(key, shape=(num_samples, ndim))

    spline = model_nf.RQSplineFlow(ndim, standardize=True)
    spline.fit(samples, epochs=epochs)

    nsamples = 10000
    spline.temperature = 1.
    flow_samples = spline.sample(nsamples)
    sample_var = jnp.var(flow_samples, axis = 0)
    spline.temperature = 0.8
    flow_samples_concentrated = spline.sample(nsamples)
    sample_var_concentrated = jnp.var(flow_samples_concentrated, axis = 0)

    for i in range(ndim):
        assert sample_var[i] > sample_var_concentrated[i], "Reducing temperature increases variance in dimension " + str(i)
    
    sample_mean = jnp.mean(flow_samples, axis=0)

    for i in range(ndim):
        assert sample_mean[i] == pytest.approx(0.0, abs = 0.1), "Sample mean in dimension " + str(i) + " is " + str(sample_mean[i])
        assert sample_var[i] == pytest.approx(1.0, abs = 0.1), "Sample variance in dimension " + str(i) + " is " + str(sample_var[i])