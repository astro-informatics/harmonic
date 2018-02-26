import evidence as cbe
import pytest
import numpy as np
from scipy.stats import kurtosis
import chains as ch
import model as md

def test_constructor():

    with pytest.raises(ValueError):
        rho = cbe.Evidence(0, 100)

    with pytest.raises(ValueError):
        rho = cbe.Evidence(100, 0)

    nchains = 100
    ndim = 1000
    
    rho = cbe.Evidence(nchains, ndim)

    assert rho.nchains        == nchains
    assert rho.p              == pytest.approx(0.0)
    assert rho.s2             == pytest.approx(0.0)
    assert rho.v2             == pytest.approx(0.0)
    assert rho.p_i.size       == nchains 
    assert rho.nsamples_per_chain.size == nchains 
    assert rho.mean_shift     == pytest.approx(0.0)
    assert rho.mean_shift_set == False
    for i_chain in range(nchains):
        assert rho.p_i[i_chain]       == pytest.approx(0.0)
        assert rho.nsamples_per_chain[i_chain] == 0

def test_set_mean_shift():
    nchains = 100
    ndim = 1000
    rho = cbe.Evidence(nchains, ndim)

    with pytest.raises(ValueError):
        rho.set_mean_shift(np.nan)

    rho.set_mean_shift(2.0)

    assert rho.mean_shift      == pytest.approx(2.0)
    assert rho.mean_shift_set  == True

def test_process_run():

    nchains = 10
    n_samples = 20
    ndim = 1000

    rho = cbe.Evidence(nchains, ndim)

    np.random.seed(1)
    samples   = np.random.randn(nchains,n_samples)
    rho.p_i       = np.sum(samples,axis=1)
    rho.nsamples_per_chain = np.ones(nchains, dtype=int)*n_samples
    rho.process_run()

    p  = np.mean(samples)
    s2 = np.std(np.sum(samples,axis=1)/n_samples)**2/(nchains)
    print(np.std(np.sum(samples,axis=1)/n_samples)**2, nchains)
    v2 = s2**2*(kurtosis(np.sum(samples,axis=1)/n_samples) + 2 + 2/(nchains-1))/nchains

    assert rho.p  == pytest.approx(p,abs=1E-7)
    assert rho.s2 == pytest.approx(s2)
    assert rho.v2 == pytest.approx(v2)

    np.random.seed(1)
    mean_shift     = 1.0
    samples_scaled = np.random.randn(nchains,n_samples)*np.exp(-mean_shift)
    samples        = samples_scaled*np.exp(mean_shift)
    rho.p_i        = np.sum(samples_scaled,axis=1)
    rho.nsamples_per_chain = np.ones(nchains, dtype=int)*n_samples
    rho.mean_shift = mean_shift
    rho.process_run()

    p  = np.mean(samples)
    s2 = np.std(np.sum(samples,axis=1)/n_samples)**2/(nchains)
    v2 = s2**2*(kurtosis(np.sum(samples,axis=1)/n_samples) + 2 + 2./(nchains-1))/nchains

    assert rho.p  == pytest.approx(p,abs=1E-7)
    assert rho.s2 == pytest.approx(s2)
    assert rho.v2 == pytest.approx(v2)

def test_add_chains():

    nchains   = 200
    nsamples  = 500
    ndim      = 2

    # create classes
    domain = [np.array([1E-1,1E1])]
    sphere = md.HyperSphere(ndim, domain)
    chain  = ch.Chains(ndim)
    cal_ev = cbe.Evidence(nchains, ndim)

    # create samples of unnormalised Gaussian
    np.random.seed(30)
    X = np.random.randn(nchains,nsamples,ndim)
    Y = -np.sum(X*X,axis=2)/2.0

    # Add samples to chains
    chain.add_chains_3d(X, Y)

    with pytest.raises(ValueError):
        cal_ev.add_chains(chain, sphere)

    # Fit the Hyper_sphere
    sphere.fit(chain.samples,chain.ln_posterior)

    
    sphere_dum = md.HyperSphere(ndim+1, domain)
    sphere_dum.fitted = True
    with pytest.raises(ValueError):
        cal_ev.add_chains(chain,sphere_dum)

    # Calculate evidence
    cal_ev.add_chains(chain,sphere)

    assert cal_ev.p       == pytest.approx(0.159438606) 
    assert cal_ev.s2      == pytest.approx(1.158805126e-07)
    assert cal_ev.v2**0.5 == pytest.approx(1.142786462e-08)

    return
