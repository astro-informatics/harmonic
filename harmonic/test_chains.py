import chains as ch
import pytest
import numpy as np

def test_constructor():
    
    ndim = 0    
    with pytest.raises(ValueError):
        chains = ch.Chains(ndim)
        
    ndim = 3    
    chains = ch.Chains(ndim)
    assert chains.ndim == ndim
    assert chains.nchains == 0
    assert chains.nsamples == 0
    assert len(chains.start_indices) == 1
    assert chains.start_indices[0] == 0


def test_add_chain():

    ndim     = 8
    nsamples1 = 1000

    chains = ch.Chains(ndim)

    np.random.seed(40)
    samples1 = np.random.randn(nsamples1,ndim)

    chains.add_chain(samples1)

    assert chains.nchains == 1
    assert chains.nsamples == nsamples1
    assert len(chains.start_indices) == 2
    assert chains.start_indices[0] == 0
    assert chains.start_indices[1] == nsamples1
    random_sample = np.random.randint(nsamples1)
    random_dim    = 4
    assert chains.samples[random_sample,random_dim] == samples1[random_sample,random_dim]

    nsamples2 = 3000
    samples2  = np.random.randn(nsamples2,ndim)
    chains.add_chain(samples2)

    assert chains.nchains == 2
    assert chains.nsamples == nsamples1 + nsamples2
    assert len(chains.start_indices) == 3
    assert chains.start_indices[0] == 0
    assert chains.start_indices[1] == nsamples1
    assert chains.start_indices[2] == nsamples1 + nsamples2
    random_sample =  nsamples1 + np.random.randint(nsamples2)
    random_dim    =  3
    assert chains.samples[random_sample,random_dim] == samples2[random_sample-nsamples1,random_dim]
    random_sample = np.random.randint(nsamples1)
    random_dim    =  7
    assert chains.samples[random_sample,random_dim] == samples1[random_sample,random_dim]

def test_get_indexes():

    ndim      = 10 
    nsamples1 = 300

    chains = ch.Chains(ndim)

    np.random.seed(40)
    samples1 = np.random.randn(nsamples1,ndim)

    chains.add_chain(samples1)

    nsamples2 = 3000
    samples2  = np.random.randn(nsamples2,ndim)
    chains.add_chain(samples2)

    with pytest.raises(ValueError):
        chains.get_chain_idexes(-1)
    
    with pytest.raises(ValueError):
        chains.get_chain_idexes(2)
    
    chain_start, chain_end = chains.get_chain_idexes(0)
    assert chain_start == 0
    assert chain_end   == nsamples1
    
    chain_start, chain_end = chains.get_chain_idexes(1)
    assert chain_start == nsamples1
    assert chain_end   == nsamples1 + nsamples2
    

    



