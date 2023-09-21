import pytest
import harmonic.model_nf as model_nf
import jax.numpy as jnp
import emcee
import numpy as np


def test_RealNVP_constructor():

    with pytest.raises(ValueError):
        RealNVP = model_nf.RealNVPModel(0)

    with pytest.raises(ValueError):
        RealNVP = model_nf.RealNVPModel(-1)

    ndim = 3
    RealNVP = model_nf.RealNVPModel(ndim) 

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

    assert RealNVP.fitted == False
    training_samples = jnp.zeros((12,ndim))
    RealNVP.fit(training_samples)
    assert RealNVP.fitted == True


def test_RQSpline_constructor():

    with pytest.raises(ValueError):
        spline = model_nf.RQSplineFlow(0)

    with pytest.raises(ValueError):
        spline = model_nf.RQSplineFlow(-1)

    ndim = 3
    spline = model_nf.RQSplineFlow(ndim) 

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

    assert spline.fitted == False
    training_samples = jnp.zeros((12,ndim))
    spline.fit(training_samples)
    assert spline.fitted == True




def test_RealNVP_temperature():

    # Define the number of dimensions and the mean of the Gaussian
    ndim = 2

    def log_prob(x):
        "Gaussian in ndim dimensions with zero mean and unit variance."
        mu = jnp.zeros(ndim)
        cov = jnp.eye(ndim)
        diff = x - mu
        return -0.5 * jnp.dot(diff, jnp.linalg.solve(cov, diff))

    # Initialize the sampler with the number of walkers and the number of dimensions
    nwalkers = 10
    burnin = 100
    nsteps = 500

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    initial_positions = np.random.randn(nwalkers, ndim)

    state = sampler.run_mcmc(initial_positions, burnin)
    sampler.reset()
    sampler.run_mcmc(state, nsteps)
    samples = sampler.get_chain().reshape((-1,ndim))

    RealNVP = model_nf.RealNVPModel(ndim)
    RealNVP.fit(samples)

    nsamples = 100
    RealNVP.temperature = 1.
    flow_samples = RealNVP.sample(nsamples)
    sample_var = jnp.var(flow_samples, axis = 0)
    RealNVP.temperature = 0.8
    flow_samples = RealNVP.sample(nsamples)
    sample_var_concentrated = jnp.var(flow_samples, axis = 0)

    for i in range(ndim):
        assert sample_var[i] > sample_var_concentrated[i], "Reducing temperature increases variance in dimension " + str(i)



def test_RQSpline_temperature():

    # Define the number of dimensions and the mean of the Gaussian
    ndim = 2

    def log_prob(x):
        "Gaussian in ndim dimensions with zero mean and unit variance."
        mu = jnp.zeros(ndim)
        cov = jnp.eye(ndim)
        diff = x - mu
        return -0.5 * jnp.dot(diff, jnp.linalg.solve(cov, diff))

    # Initialize the sampler with the number of walkers and the number of dimensions
    nwalkers = 10
    burnin = 100
    nsteps = 500

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    initial_positions = np.random.randn(nwalkers, ndim)

    state = sampler.run_mcmc(initial_positions, burnin)
    sampler.reset()
    sampler.run_mcmc(state, nsteps)
    samples = sampler.get_chain().reshape((-1,ndim))

    spline = model_nf.RQSplineFlow(ndim)
    spline.fit(samples)

    nsamples = 100
    spline.temperature = 1.
    flow_samples = spline.sample(nsamples)
    sample_var = jnp.var(flow_samples, axis = 0)
    spline.temperature = 0.8
    flow_samples = spline.sample(nsamples)
    sample_var_concentrated = jnp.var(flow_samples, axis = 0)

    for i in range(ndim):
        assert sample_var[i] > sample_var_concentrated[i], "Reducing temperature increases variance in dimension " + str(i)