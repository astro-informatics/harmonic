import numpy as np
import data_handle as dh
import chains as ch

def test_split_data():

    ndim        = 5
    nsamples    = 100
    nchains     = 200
    split_ratio = 0.5

    chains_all = ch.Chains(ndim)

    np.random.seed(3)
    samples       = np.random.randn(nchains, nsamples, ndim)
    ln_posterior  = -np.sum(samples*samples, axis=2)/2.0

    chains_all.add_chains_3d(samples, ln_posterior)

    chains_train, chains_use = dh.split_data(chains_all, split_ratio=split_ratio)


    nchains_train = int(nchains*split_ratio)

    assert chains_train.nchains            == nchains_train
    assert chains_train.nsamples           == nsamples * nchains_train
    assert len(chains_train.start_indices) == nchains_train + 1
    for i in range(nchains_train + 1):
        assert chains_train.start_indices[i] == i * nsamples

    assert chains_train.samples.shape[0]        == nsamples * nchains_train
    assert chains_train.samples.shape[1]        == ndim
    assert chains_train.ln_posterior.shape[0]   == nsamples * nchains_train

    random_sample = np.random.randint(nsamples * nchains_train)
    random_dim    = 3
    assert chains_train.samples[random_sample,random_dim] \
        == samples[random_sample // nsamples,
                    random_sample % nsamples,random_dim]

    nchains_use = nchains - nchains_train

    assert chains_use.nchains      == nchains_use
    assert chains_use.nsamples     == nsamples * nchains_use
    assert len(chains_use.start_indices) == nchains_use + 1
    for i in range(nchains_use + 1):
        assert chains_use.start_indices[i] == i * nsamples

    assert chains_use.samples.shape[0]        == nsamples * nchains_use
    assert chains_use.samples.shape[1]        == ndim
    assert chains_use.ln_posterior.shape[0]   == nsamples * nchains_use

    random_sample = np.random.randint(nsamples * nchains_use)
    random_dim    = 3
    assert chains_use.samples[random_sample,random_dim] \
        == samples[(random_sample // nsamples) + nchains_train,
                    random_sample % nsamples,random_dim]


    ndim        = 5
    nsamples    = 100
    nchains     = 200
    split_ratio = 0.75

    chains_all = ch.Chains(ndim)

    np.random.seed(3)
    samples       = np.random.randn(nchains, nsamples, ndim)
    ln_posterior  = -np.sum(samples*samples, axis=2)/2.0

    chains_all.add_chains_3d(samples, ln_posterior)

    chains_train, chains_use = dh.split_data(chains_all, split_ratio=split_ratio)


    nchains_train = int(nchains*split_ratio)

    assert chains_train.nchains            == nchains_train
    assert chains_train.nsamples           == nsamples * nchains_train
    assert len(chains_train.start_indices) == nchains_train + 1
    for i in range(nchains_train + 1):
        assert chains_train.start_indices[i] == i * nsamples

    assert chains_train.samples.shape[0]        == nsamples * nchains_train
    assert chains_train.samples.shape[1]        == ndim
    assert chains_train.ln_posterior.shape[0]   == nsamples * nchains_train

    random_sample = np.random.randint(nsamples * nchains_train)
    random_dim    = 3
    assert chains_train.samples[random_sample,random_dim] \
        == samples[random_sample // nsamples,
                    random_sample % nsamples,random_dim]

    nchains_use = nchains - nchains_train

    assert chains_use.nchains      == nchains_use
    assert chains_use.nsamples     == nsamples * nchains_use
    assert len(chains_use.start_indices) == nchains_use + 1
    for i in range(nchains_use + 1):
        assert chains_use.start_indices[i] == i * nsamples

    assert chains_use.samples.shape[0]        == nsamples * nchains_use
    assert chains_use.samples.shape[1]        == ndim
    assert chains_use.ln_posterior.shape[0]   == nsamples * nchains_use

    random_sample = np.random.randint(nsamples * nchains_use)
    random_dim    = 3
    assert chains_use.samples[random_sample,random_dim] \
        == samples[(random_sample // nsamples) + nchains_train,
                    random_sample % nsamples,random_dim]


def test_cross_validation():

    ndim        = 2
    nsamples    = 100
    nchains     = 200
    ncross      = 5

    hyper_parameters = [10**R for R in range(-5,0)]

    print(hyper_parameters)

    chains = ch.Chains(ndim)

    np.random.seed(3)
    samples       = np.random.randn(nchains, nsamples, ndim)
    ln_posterior  = -np.sum(samples*samples, axis=2)/2.0

    chains.add_chains_3d(samples, ln_posterior)

    dh.cross_validation(chains, [], hyper_parameters)

test_cross_validation()