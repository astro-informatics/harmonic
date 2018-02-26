import evidence as cbe
import pytest
import numpy as np
from scipy.stats import kurtosis
import chains as ch
import model as md

def test_constructor():
    
    nchains = 100
    ndim = 1000
    domain = [np.array([1E-1,1E1])]
    
    sphere = md.HyperSphere(ndim, domain)
    
    sphere.fitted = False    
    with pytest.raises(ValueError):
        rho = cbe.Evidence(nchains=100, model=sphere)

    sphere.fitted = True
    with pytest.raises(ValueError):
        rho = cbe.Evidence(nchains=0, model=sphere)
    
    rho = cbe.Evidence(nchains, sphere)

    assert rho.nchains          == nchains
    assert rho.evidence_inv         == pytest.approx(0.0)
    assert rho.evidence_inv_var     == pytest.approx(0.0)
    assert rho.evidence_inv_var_var == pytest.approx(0.0)
    assert rho.running_sum.size         == nchains 
    assert rho.nsamples_per_chain.size == nchains 
    assert rho.mean_shift     == pytest.approx(0.0)
    assert rho.mean_shift_set == False
    for i_chain in range(nchains):
        assert rho.running_sum[i_chain]       == pytest.approx(0.0)
        assert rho.nsamples_per_chain[i_chain] == 0

def test_set_mean_shift():
    
    nchains = 100
    ndim = 1000    
    domain = [np.array([1E-1,1E1])]
    sphere = md.HyperSphere(ndim, domain)
    sphere.fitted = True
    rho = cbe.Evidence(nchains, sphere)
    with pytest.raises(ValueError):
        rho.set_mean_shift(np.nan)

    rho.set_mean_shift(2.0)

    assert rho.mean_shift      == pytest.approx(2.0)
    assert rho.mean_shift_set  == True

def test_process_run():

    nchains = 10
    n_samples = 20
    ndim = 1000

    domain = [np.array([1E-1,1E1])]
    sphere = md.HyperSphere(ndim, domain)
    sphere.fitted = True
    rho = cbe.Evidence(nchains, sphere)

    np.random.seed(1)
    samples   = np.random.randn(nchains,n_samples)
    rho.running_sum       = np.sum(samples,axis=1)
    rho.nsamples_per_chain = np.ones(nchains, dtype=int)*n_samples
    rho.process_run()

    evidence_inv  = np.mean(samples)
    evidence_inv_var = np.std(np.sum(samples,axis=1)/n_samples)**2/(nchains)
    # print(np.std(np.sum(samples,axis=1)/n_samples)**2, nchains)
    evidence_inv_var_var = evidence_inv_var**2*(kurtosis(np.sum(samples,axis=1)/n_samples) + 2 + 2/(nchains-1))/nchains

    assert rho.evidence_inv == pytest.approx(evidence_inv,abs=1E-7)
    assert rho.evidence_inv_var == pytest.approx(evidence_inv_var)
    assert rho.evidence_inv_var_var == pytest.approx(evidence_inv_var_var)

    np.random.seed(1)
    post           = np.random.uniform(high=1E3, size=(nchains,n_samples))
    samples        = 1.0/post
    mean_shift     = np.mean(np.log(post))
    samples_scaled = samples*np.exp(mean_shift)
    rho.running_sum        = np.sum(samples_scaled,axis=1)
    rho.nsamples_per_chain = np.ones(nchains, dtype=int)*n_samples
    rho.mean_shift = mean_shift
    rho.process_run()

    evidence_inv  = np.mean(samples)
    evidence_inv_var = np.std(np.sum(samples,axis=1)/n_samples)**2/(nchains)
    evidence_inv_var_var = evidence_inv_var**2*(kurtosis(np.sum(samples,axis=1)/n_samples) + 2 + 2./(nchains-1))/nchains

    assert rho.evidence_inv  == pytest.approx(evidence_inv,abs=1E-7)
    assert rho.evidence_inv_var == pytest.approx(evidence_inv_var)
    assert rho.evidence_inv_var_var == pytest.approx(evidence_inv_var_var)

def test_add_chains():

    nchains   = 200
    nsamples  = 500
    ndim      = 2

    # Create samples of unnormalised Gaussian    
    np.random.seed(30)
    X = np.random.randn(nchains,nsamples,ndim)
    Y = -np.sum(X*X,axis=2)/2.0

    # Add samples to chains
    chain  = ch.Chains(ndim)    
    chain.add_chains_3d(X, Y)

    # Fit the Hyper_sphere
    domain = [np.array([1E-1,1E1])]
    sphere = md.HyperSphere(ndim, domain)        
    sphere.fit(chain.samples, chain.ln_posterior)    

    # Calculate evidence
    cal_ev = cbe.Evidence(nchains, sphere)
    cal_ev.add_chains(chain)

    assert cal_ev.evidence_inv              == pytest.approx(0.159438606) 
    assert cal_ev.evidence_inv_var          == pytest.approx(1.158805126e-07)
    assert cal_ev.evidence_inv_var_var**0.5 == pytest.approx(1.142786462e-08)

    return

