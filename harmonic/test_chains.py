import chains as ch
import pytest
import numpy as np

def test_constructor():
    
    ndim = 0    
    with pytest.raises(ValueError):
        chains = ch.Chains(ndim)
        
    ndim = 3    
    chains = ch.Chains(ndim)
    assert chains.ndim  == ndim
    assert chains.nchains == 0
    assert chains.nsamples == 0
    assert len(chains.start_indices) == 1
    assert chains.start_indices[0] == 0
    assert chains.samples.shape[0] == 0
    assert chains.samples.shape[1] == ndim
    assert chains.ln_posterior.shape[0] == 0

def test_add_chain():

    ndim = 8
    nsamples1 = 1000

    chains = ch.Chains(ndim)

    # Check cannot add samples with different ndim.
    ndim_tmp = 9
    with pytest.raises(ValueError):
        chains.add_chain(np.zeros((2,ndim_tmp)), np.zeros(2))

    with pytest.raises(ValueError):
        chains.add_chain(np.zeros((2,ndim)), np.zeros(3))

    # Add random samples1.
    np.random.seed(40)
    samples1 = np.random.randn(nsamples1,ndim)
    ln_posterior1 = np.random.randn(nsamples1)

    chains.add_chain(samples1, ln_posterior1)

    assert chains.nchains == 1
    assert chains.nsamples == nsamples1
    assert len(chains.start_indices) == 2
    assert chains.start_indices[0] == 0
    assert chains.start_indices[1] == nsamples1
    assert chains.samples.shape[0] == nsamples1
    assert chains.samples.shape[1] == ndim
    assert chains.ln_posterior.shape[0] == nsamples1

    random_sample = np.random.randint(nsamples1)
    random_dim = 4
    assert chains.samples[random_sample,random_dim]  \
        == samples1[random_sample,random_dim]
    assert chains.ln_posterior[random_sample]        \
        == ln_posterior1[random_sample]

    # Add random samples2
    nsamples2 = 3000
    samples2      = np.random.randn(nsamples2,ndim)
    ln_posterior2 = np.random.randn(nsamples2)
    chains.add_chain(samples2, ln_posterior2)

    assert chains.nchains == 2
    assert chains.nsamples == nsamples1 + nsamples2
    assert len(chains.start_indices) == 3
    assert chains.start_indices[0] == 0
    assert chains.start_indices[1] == nsamples1
    assert chains.start_indices[2] == nsamples1 + nsamples2
    assert chains.samples.shape[0] == nsamples1 + nsamples2
    assert chains.samples.shape[1] == ndim
    assert chains.ln_posterior.shape[0] == nsamples1 + nsamples2
 
    random_sample =  nsamples1 + np.random.randint(nsamples2)
    random_dim =  3
    assert chains.samples[random_sample,random_dim]  \
        == samples2[random_sample-nsamples1,random_dim]
    assert chains.ln_posterior[random_sample]        \
        == ln_posterior2[random_sample-nsamples1]
    random_sample = np.random.randint(nsamples1)
    random_dim    =  7
    assert chains.samples[random_sample,random_dim] \
        == samples1[random_sample,random_dim]
    assert chains.ln_posterior[random_sample]       \
        == ln_posterior1[random_sample]

def test_add_chains_2d_and_copy():

    ndim = 8
    nsamples1 = 100
    nchains1 = 60

    chains = ch.Chains(ndim)

    # Set up samples1.
    np.random.seed(50)
    samples1 = np.random.randn(nsamples1 * nchains1, ndim)
    ln_posterior1 = np.random.randn(nsamples1 * nchains1)

    # Check cannot add samples with different ndim.
    with pytest.raises(ValueError):
        chains.add_chains_2d(np.zeros((2,ndim+1)), np.zeros(2),1)
    with pytest.raises(ValueError):
        chains.add_chains_2d(samples1, ln_posterior1, nchains1+1)
    with pytest.raises(ValueError):
        chains.add_chains_2d(np.zeros((2,ndim)), np.zeros(3),1)

    chains.add_chains_2d(samples1, ln_posterior1, nchains1)

    assert chains.nchains == nchains1
    assert chains.nsamples == nsamples1 * nchains1
    assert len(chains.start_indices) == nchains1 + 1
    for i in range(nchains1+1):
        assert chains.start_indices[i] == i * nsamples1
    assert chains.samples.shape[0] == nsamples1 * nchains1
    assert chains.samples.shape[1] == ndim
    assert chains.ln_posterior.shape[0] == nsamples1 * nchains1

    random_sample = np.random.randint(nsamples1 * nchains1)
    random_dim    = 0
    assert chains.samples[random_sample,random_dim] \
        == samples1[random_sample,random_dim]
    assert chains.ln_posterior[random_sample] \
        == ln_posterior1[random_sample]

    nsamples2 = 100
    nchains2  = 300

    samples2 = np.random.randn(nsamples2*nchains2, ndim)
    ln_posterior2 = np.random.randn(nsamples2*nchains2)

    chains.add_chains_2d(samples2, ln_posterior2, nchains2)

    assert chains.nchains == nchains1 + nchains2
    assert chains.nsamples == nsamples1 * nchains1 + nsamples2 * nchains2 
    assert len(chains.start_indices) == nchains1 + nchains2 + 1
    for i in range(nchains1):
        assert chains.start_indices[i] == i*nsamples1
    for i in range(nchains1,nchains2+1):
        assert chains.start_indices[i+nchains1] \
            == nchains1 * nsamples1 + i * nsamples2
    assert chains.samples.shape[0] \
        == nsamples1 * nchains1 + nsamples2 * nchains2
    assert chains.samples.shape[1] == ndim
    assert chains.ln_posterior.shape[0] \
        == nsamples1 * nchains1 + nsamples2 * nchains2

    random_sample = np.random.randint(nsamples1)
    random_dim =  5
    assert chains.samples[random_sample,random_dim] \
        == samples1[random_sample,random_dim]
    assert chains.ln_posterior[random_sample]       \
        == ln_posterior1[random_sample]
    random_sample = nsamples1*nchains1 \
        + np.random.randint(nsamples2*nchains2)
    random_dim = 2
    assert chains.samples[random_sample,random_dim] \
        == samples2[random_sample-nsamples1*nchains1,random_dim]
    assert chains.ln_posterior[random_sample]       \
        == ln_posterior2[random_sample-nsamples1*nchains1]

    chains2 = chains.copy()

    assert chains2.nchains  == nchains1 + nchains2
    assert chains2.nsamples \
        == nsamples1 * nchains1 + nsamples2 * nchains2 
    assert len(chains2.start_indices) == nchains1 + nchains2 + 1

    for i in range(nchains1):
        assert chains2.start_indices[i] == i * nsamples1
    for i in range(nchains1,nchains2+1):
        assert chains.start_indices[i+nchains1] \
            == nchains1*nsamples1 + i*nsamples2
    assert chains2.samples.shape[0] \
        == nsamples1*nchains1 + nsamples2*nchains2
    assert chains2.samples.shape[1] == ndim
    assert chains2.ln_posterior.shape[0] \
        == nsamples1*nchains1 + nsamples2*nchains2

    random_sample = np.random.randint(nsamples1)
    random_dim =  5
    assert chains2.samples[random_sample,random_dim] \
        == samples1[random_sample,random_dim]
    assert chains2.ln_posterior[random_sample]       \
        == ln_posterior1[random_sample]
    random_sample = nsamples1*nchains1 \
        + np.random.randint(nsamples2*nchains2)
    random_dim    = 2
    assert chains2.samples[random_sample,random_dim] \
        == samples2[random_sample-nsamples1*nchains1,random_dim]
    assert chains2.ln_posterior[random_sample] \
        == ln_posterior2[random_sample-nsamples1*nchains1]

def test_add_chains_2d_list():

    ndim = 8
    nsamples1 = 100
    nchains1 = 60

    chains_list1 = []
    for i_chain in range(nchains1):
        chains_list1.append(i_chain*nsamples1)
    chains_list1.append(nchains1*nsamples1)

    chains = ch.Chains(ndim)

    # Set up samples1.
    np.random.seed(50)
    samples1 = np.random.randn(nsamples1 * nchains1, ndim)
    ln_posterior1 = np.random.randn(nsamples1 * nchains1)

    # Check cannot add samples with different ndim.
    with pytest.raises(ValueError):
        chains.add_chains_2d_list(np.zeros((2,ndim+1)), 
                                  np.zeros(2),1,[0,1])
    with pytest.raises(ValueError):
        chains.add_chains_2d_list(samples1, 
                                  ln_posterior1, 
                                  nchains1+1, 
                                  [0 for i in range(nchains1)])
    with pytest.raises(ValueError):
        chains.add_chains_2d_list(np.zeros((2,ndim)), 
                                  np.zeros(3), 1, [1,2])

    chains.add_chains_2d_list(samples1, 
                              ln_posterior1, 
                              nchains1, 
                              chains_list1)

    assert chains.nchains == nchains1
    assert chains.nsamples == nsamples1 * nchains1
    assert len(chains.start_indices) == nchains1 + 1
    for i in range(nchains1+1):
        assert chains.start_indices[i] == i * nsamples1
    assert chains.samples.shape[0] == nsamples1 * nchains1
    assert chains.samples.shape[1] == ndim
    assert chains.ln_posterior.shape[0] == nsamples1 * nchains1

    random_sample = np.random.randint(nsamples1 * nchains1)
    random_dim    = 0
    assert chains.samples[random_sample,random_dim] \
        == samples1[random_sample,random_dim]
    assert chains.ln_posterior[random_sample] \
        == ln_posterior1[random_sample]

    nsamples2 = 100
    nchains2  = 300

    samples2 = np.random.randn(nsamples2*nchains2, ndim)
    ln_posterior2 = np.random.randn(nsamples2*nchains2)

    chains_list2 = []
    for i_chain in range(nchains2):
        chains_list2.append(i_chain*nsamples2)
    chains_list2.append(nchains2*nsamples2)

    chains.add_chains_2d_list(samples2, 
                              ln_posterior2, 
                              nchains2, 
                              chains_list2)

    assert chains.nchains == nchains1 + nchains2
    assert chains.nsamples \
        == nsamples1 * nchains1 + nsamples2 * nchains2 
    assert len(chains.start_indices) == nchains1 + nchains2 + 1
    for i in range(nchains1):
        assert chains.start_indices[i] == i*nsamples1
    for i in range(nchains1,nchains2+1):
        assert chains.start_indices[i+nchains1] \
            == nchains1 * nsamples1 + i * nsamples2
    assert chains.samples.shape[0] \
        == nsamples1 * nchains1 + nsamples2 * nchains2
    assert chains.samples.shape[1] == ndim
    assert chains.ln_posterior.shape[0] \
        == nsamples1 * nchains1 + nsamples2 * nchains2

    random_sample = np.random.randint(nsamples1)
    random_dim =  5
    assert chains.samples[random_sample,random_dim] \
        == samples1[random_sample,random_dim]
    assert chains.ln_posterior[random_sample]       \
        == ln_posterior1[random_sample]
    random_sample = nsamples1*nchains1 \
        + np.random.randint(nsamples2*nchains2)
    random_dim = 2
    assert chains.samples[random_sample,random_dim] \
        == samples2[random_sample-nsamples1*nchains1,random_dim]
    assert chains.ln_posterior[random_sample]       \
        == ln_posterior2[random_sample-nsamples1*nchains1]     

def test_add_chains_3d():

    ndim = 5
    nsamples1 = 100
    nchains1 = 50

    chains = ch.Chains(ndim)

    np.random.seed(3)
    samples1 = np.random.randn(nchains1 ,nsamples1, ndim)
    ln_posterior1 = np.random.randn(nchains1 ,nsamples1)

    # Check cannot add sampes with different ndim
    with pytest.raises(ValueError):
        chains.add_chains_3d(np.zeros((2,2,ndim+1)),np.zeros((2,2)))
    with pytest.raises(ValueError):
        chains.add_chains_3d(np.zeros((2,2,ndim)),np.zeros((1,2)))
    with pytest.raises(ValueError):
        chains.add_chains_3d(np.zeros((2,2,ndim)),np.zeros((2,1)))

    chains.add_chains_3d(samples1, ln_posterior1)

    assert chains.nchains  == nchains1
    assert chains.nsamples == nsamples1 * nchains1
    assert len(chains.start_indices) == nchains1 + 1
    for i in range(nchains1 + 1):
        assert chains.start_indices[i] == i * nsamples1
    assert chains.samples.shape[0]        == nsamples1 * nchains1
    assert chains.samples.shape[1]        == ndim
    assert chains.ln_posterior.shape[0]   == nsamples1 * nchains1

    random_sample = np.random.randint(nsamples1 * nchains1)
    random_dim = 3
    assert chains.samples[random_sample,random_dim] \
        == samples1[random_sample // nsamples1,
                    random_sample % nsamples1,random_dim]
    assert chains.ln_posterior[random_sample] \
        == ln_posterior1[random_sample // nsamples1, \
                         random_sample % nsamples1]

    nsamples2 = 100
    nchains2 = 300

    samples2 = np.random.randn(nchains2,nsamples2,ndim)
    ln_posterior2 = np.random.randn(nchains2,nsamples2)

    # Add samples2
    chains.add_chains_3d(samples2, ln_posterior2)

    # Checks after added second set of chains.
    assert chains.nchains  == nchains1 + nchains2
    assert chains.nsamples == nsamples1 * nchains1 + nsamples2 * nchains2 
    assert len(chains.start_indices) == nchains1 + nchains2 + 1
    for i in range(nchains1):
        assert chains.start_indices[i] == i*nsamples1
    for i in range(nchains1,nchains2+1):
        assert chains.start_indices[i + nchains1] \
            == nchains1 * nsamples1 + i * nsamples2

    assert chains.samples.shape[0] \
        == nsamples1 * nchains1 + nsamples2 * nchains2
    assert chains.samples.shape[1] == ndim
    assert chains.ln_posterior.shape[0] \
        == nsamples1 * nchains1 + nsamples2 * nchains2

    random_sample = np.random.randint(nsamples1)
    random_dim    =  4
    assert chains.samples[random_sample,random_dim] \
        == samples1[random_sample // nsamples1,
                    random_sample % nsamples1,random_dim]
    assert chains.ln_posterior[random_sample] \
        == ln_posterior1[random_sample // nsamples1,
                         random_sample % nsamples1]
    random_sample_sub = np.random.randint(nsamples2 * nchains2)
    random_sample = nsamples1 * nchains1 + random_sample_sub
    random_dim = 2
    assert chains.samples[random_sample,random_dim] \
        == samples2[random_sample_sub//nsamples2,
                    random_sample_sub%nsamples2,
                    random_dim]
    assert chains.ln_posterior[random_sample] \
        == ln_posterior2[random_sample_sub // nsamples2,
                         random_sample_sub % nsamples2]

def test_get_sub_chains():

    ndim     = 4
    nsamples = 100
    nchains  = 10

    chains_wanted = [9,5,6]

    chains = ch.Chains(ndim)

    np.random.seed(40)
    samples = np.random.randn(nchains,nsamples,ndim)
    ln_posterior = np.random.randn(nchains,nsamples)

    chains.add_chains_3d(samples, ln_posterior)

    with pytest.raises(ValueError):
        chains.get_sub_chains([0,2,-1])
    with pytest.raises(ValueError):
        chains.get_sub_chains([0,2,nchains+1])

    sub_chains = chains.get_sub_chains(chains_wanted)

    assert sub_chains.nchains == len(chains_wanted)
    random_sub_chain = 2
    random_chain     = chains_wanted[random_sub_chain]
    random_sample    = np.random.randint(nsamples)
    random_dim       = 3
    assert sub_chains.samples[random_sub_chain * nsamples + random_sample,\
                              random_dim] \
                              == samples[random_chain,random_sample,random_dim]

def test_get_indexes():

    ndim = 10 
    nsamples1 = 300

    chains = ch.Chains(ndim)

    np.random.seed(40)
    samples1 = np.random.randn(nsamples1,ndim)
    ln_posterior1 = np.random.randn(nsamples1)

    chains.add_chain(samples1, ln_posterior1)

    nsamples2 = 3000
    samples2 = np.random.randn(nsamples2,ndim)
    ln_posterior2 = np.random.randn(nsamples2)
    chains.add_chain(samples2, ln_posterior2)

    with pytest.raises(ValueError):
        chains.get_chain_indices(-1)
    
    with pytest.raises(ValueError):
        chains.get_chain_indices(2)
    
    chain_start, chain_end = chains.get_chain_indices(0)
    assert chain_start == 0
    assert chain_end == nsamples1
    
    chain_start, chain_end = chains.get_chain_indices(1)
    assert chain_start == nsamples1
    assert chain_end == nsamples1 + nsamples2
    
def test_add():
    
    ndim = 8
    np.random.seed(50)
    
    # Set up first Chains object.
    chains1 = ch.Chains(ndim)    
    nsamples1 = 100
    nchains1 = 60
    samples1 = np.random.randn(nsamples1 * nchains1, ndim)
    ln_posterior1 = np.random.randn(nsamples1 * nchains1)
    chains1.add_chains_2d(samples1, ln_posterior1, nchains1)

    # Set up second Chains object.
    chains2 = ch.Chains(ndim)
    nsamples2 = 100
    nchains2 = 300
    samples2 = np.random.randn(nsamples2 * nchains2, ndim)
    ln_posterior2 = np.random.randn(nsamples2 * nchains2)
    chains2.add_chains_2d(samples2, ln_posterior2, nchains2)

    with pytest.raises(ValueError):
        chains1.add(ch.Chains(ndim+1))

    # Copy chain1 and then add chains2.
    chains_added = chains1.copy()
    chains_added.add(chains2)
    
    # Checks on added object.
    assert chains_added.nchains == nchains1 + nchains2
    assert chains_added.nsamples == nsamples1 * nchains1 + nsamples2 * nchains2 
    assert len(chains_added.start_indices) == nchains1 + nchains2 + 1
    for i in range(nchains1):
        assert chains_added.start_indices[i] == i * nsamples1
    for i in range(nchains1, nchains2 + 1):
        assert chains_added.start_indices[i+nchains1] \
            == nchains1 * nsamples1 + i * nsamples2
    random_sample = np.random.randint(nsamples1)
    random_dim = 6
    assert chains_added.samples[random_sample,random_dim] \
        == samples1[random_sample,random_dim]
    assert chains_added.ln_posterior[random_sample] \
        == ln_posterior1[random_sample]        
    random_sample = nsamples1 * nchains1 \
        + np.random.randint(nsamples2 * nchains2)
    random_dim = 3
    assert chains_added.samples[random_sample,random_dim] \
        == samples2[random_sample-nsamples1*nchains1,random_dim]
    assert chains_added.ln_posterior[random_sample] \
        == ln_posterior2[random_sample-nsamples1*nchains1]
                

def test_nsamples_per_chain():

    ndim = 8
    np.random.seed(50)

    # Set up first Chains object.
    chains1 = ch.Chains(ndim)    
    nsamples1 = 100
    nchains1 = 60
    samples1 = np.random.randn(nsamples1 * nchains1, ndim)
    ln_posterior1 = np.random.randn(nsamples1 * nchains1)
    chains1.add_chains_2d(samples1, ln_posterior1, nchains1)

    # Set up second Chains object.
    chains2 = ch.Chains(ndim)
    nsamples2 = 120
    nchains2 = 300
    samples2 = np.random.randn(nsamples2 * nchains2, ndim)
    ln_posterior2 = np.random.randn(nsamples2 * nchains2)
    chains2.add_chains_2d(samples2, ln_posterior2, nchains2)

    # Copy chain1 and then add chains2.
    chains_added = chains1.copy()
    chains_added.add(chains2)
    
    nsamples_per_chain = chains_added.nsamples_per_chain()

    assert len(nsamples_per_chain) == nchains1 + nchains2
    for i in range(nchains1):
        assert nsamples_per_chain[i] == nsamples1
    for i in range(nchains1,nchains2+1):
        assert nsamples_per_chain[i] == nsamples2

def test_remove_burnin():
    
    ndim = 8    
    nsamples1 = 441
    nsamples2 = 169

    chains = ch.Chains(ndim)    
    np.random.seed(40)
    samples1 = np.random.randn(nsamples1, ndim)
    ln_posterior1 = np.random.randn(nsamples1)
    chains.add_chain(samples1, ln_posterior1)    
        
    samples2 = np.random.randn(nsamples2, ndim)
    ln_posterior2 = np.random.randn(nsamples2)
    chains.add_chain(samples2, ln_posterior2) 
    
    with pytest.raises(ValueError):
        chains.remove_burnin(1000)
    with pytest.raises(ValueError):
        chains.remove_burnin(200)
        
    nburn = 100
    chains.remove_burnin(nburn)
    
    assert chains.nsamples == nsamples1 + nsamples2 - 2*nburn
    nsamples_per_chain = chains.nsamples_per_chain()    
    assert nsamples_per_chain[0] == nsamples1 - nburn
    assert nsamples_per_chain[1] == nsamples2 - nburn
                
    random_sample = np.random.randint(nsamples1 - nburn)
    random_dim = 5
    assert chains.samples[random_sample, random_dim] \
        == samples1[random_sample + nburn, random_dim]
    assert chains.ln_posterior[random_sample] \
        == ln_posterior1[random_sample + nburn]        
            
    random_sample = np.random.randint(nsamples2 - nburn)
    random_dim = 2
    assert chains.samples[random_sample + nsamples1 - nburn, random_dim] \
        == samples2[random_sample + nburn, random_dim]
    assert chains.ln_posterior[random_sample + nsamples1 - nburn] \
        == ln_posterior2[random_sample + nburn]
                
def test_split_into_blocks():

    ndim = 8
    nsamples1 = 169
    nsamples2 = 441
    nsamples3 = 208

    chains = ch.Chains(ndim)    
    np.random.seed(40)
    samples1 = np.random.randn(nsamples1, ndim)
    ln_posterior1 = np.random.randn(nsamples1)
    chains.add_chain(samples1, ln_posterior1)    
        
    samples2 = np.random.randn(nsamples2, ndim)
    ln_posterior2 = np.random.randn(nsamples2)
    chains.add_chain(samples2, ln_posterior2)    
        
    samples3 = np.random.randn(nsamples3, ndim)
    ln_posterior3 = np.random.randn(nsamples3)
    chains.add_chain(samples3, ln_posterior3)            
    
    chains_blocked = chains.copy()

    with pytest.raises(ValueError):
        chains_blocked.split_into_blocks(2)

    nblocks = 10                
    chains_blocked.split_into_blocks(nblocks)
    assert chains_blocked.nchains == nblocks
    
    # Check mean number of samples per (blocked) chain is similar to desired 
    # value of chains_blocked.nsamples / nblocks.
    mean_samples_per_chain = np.mean(chains_blocked.nsamples_per_chain())     
    err = np.absolute(mean_samples_per_chain \
        - chains_blocked.nsamples / nblocks)
    assert err / mean_samples_per_chain < 0.05    
        