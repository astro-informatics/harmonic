import numpy as np
import data_handle as dh
import chains as ch
import pytest

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

    with pytest.raises(ValueError):
        chains_train, chains_use = dh.split_data(chains_all, split_ratio=-0.1)

    with pytest.raises(ValueError):
        chains_train, chains_use = dh.split_data(chains_all, split_ratio=1.5)

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


def test_validation_fit_indexes():

    nchains = 10
    ncross  = 3

    np.random.seed(0)
    indexes = list(np.random.permutation(nchains)) # creates [2, 8, 4, 9, 1, 6, 7, 3, 0, 5]

    nchains_in_val_set = nchains/ncross

    with pytest.raises(ValueError):
        dh.validation_fit_indexes(ncross, nchains_in_val_set, ncross, indexes)
    with pytest.raises(ValueError):
        dh.validation_fit_indexes(-1, nchains_in_val_set, ncross, indexes)

    indexes_val, indexes_fit = dh.validation_fit_indexes(0, nchains_in_val_set, ncross, indexes)
    assert len(indexes_val) == 3
    for index_val, index_check in zip(indexes_val, [2, 8, 4]):
        assert index_val == index_check
    for index_fit, index_check in zip(indexes_fit, [9, 1, 6, 7, 3, 0, 5]):
        assert index_fit == index_check

    indexes_val, indexes_fit = dh.validation_fit_indexes(1, nchains_in_val_set, ncross, indexes)
    assert len(indexes_val) == 3
    for index_val, index_check in zip(indexes_val, [9, 1, 6]):
        assert index_val == index_check
    for index_fit, index_check in zip(indexes_fit, [2, 8, 4, 7, 3, 0, 5]):
        assert index_fit == index_check

    indexes_val, indexes_fit = dh.validation_fit_indexes(2, nchains_in_val_set, ncross, indexes)
    assert len(indexes_val) == 4
    for index_val, index_check in zip(indexes_val, [7, 3, 0, 5]):
        assert index_val == index_check
    for index_fit, index_check in zip(indexes_fit, [2, 8, 4, 9, 1, 6]):
        assert index_fit == index_check


def test_cross_validation():

    ndim        = 2
    nsamples    = 10
    nchains     = 200
    ncross      = 2
    step        = 0

    hyper_parameters_HS  = [None for R in range(-ncross-step,-step)]
    hyper_parameters_KDE = [[10**R] for R in range(-ncross-step,-step)]

    chains = ch.Chains(ndim)

    np.random.seed(3)
    samples       = np.random.randn(nchains, nsamples, ndim)
    ln_posterior  = -np.sum(samples*samples, axis=2)/2.0

    chains.add_chains_3d(samples, ln_posterior)

    with pytest.raises(ValueError):
        dh.cross_validation(chains, [], hyper_parameters_KDE, MODEL="not_a_model")


    # just checks the result of the code is unchanged
    validation_variences = dh.cross_validation(chains, [np.array([1E-1,1E1])], \
                        hyper_parameters_HS, MODEL="HyperSphere", verbose=False)
    assert validation_variences[0] == pytest.approx(1.48812772e-05) 
    assert validation_variences[1] == pytest.approx(1.48812772e-05) 
    validation_variences = dh.cross_validation(chains, [], hyper_parameters_KDE, \
                        verbose=False)
    assert validation_variences[0] == pytest.approx(9.74522749e-05) 
    assert validation_variences[1] == pytest.approx(2.57373056e-06) 

