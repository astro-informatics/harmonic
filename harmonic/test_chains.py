import chains

def test_constructor():
    
    ndim = 0    
    with pytest.raises(ValueError):
        chains = Chains(ndim)
        
    ndim = 3    
    chains = Chains(ndim)
    assert chains.ndim == ndim
    assert chains.nchains == 0
    assert chains.nsamples == 0
    assert len(chains.start_indices) == 1
    assert chains.start_indices[0] == 0
    
    
    
        
    
    
    
    
    



