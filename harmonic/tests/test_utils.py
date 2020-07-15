import numpy as np
import harmonic.utils as utils
import harmonic.chains as ch
import harmonic.model as md
import pytest

def test_split_data():

    ndim        = 5
    nsamples    = 100
    nchains     = 200
    training_proportion = 0.5

    chains_all = ch.Chains(ndim)

    np.random.seed(3)
    samples       = np.random.randn(nchains, nsamples, ndim)
    ln_posterior  = -np.sum(samples*samples, axis=2)/2.0

    chains_all.add_chains_3d(samples, ln_posterior)

    chains_train, chains_test = utils.split_data(chains_all, training_proportion=training_proportion)

    nchains_train = int(nchains*training_proportion)

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

    nchains_test = nchains - nchains_train

    assert chains_test.nchains      == nchains_test
    assert chains_test.nsamples     == nsamples * nchains_test
    assert len(chains_test.start_indices) == nchains_test + 1
    for i in range(nchains_test + 1):
        assert chains_test.start_indices[i] == i * nsamples

    assert chains_test.samples.shape[0]        == nsamples * nchains_test
    assert chains_test.samples.shape[1]        == ndim
    assert chains_test.ln_posterior.shape[0]   == nsamples * nchains_test

    random_sample = np.random.randint(nsamples * nchains_test)
    random_dim    = 3
    assert chains_test.samples[random_sample,random_dim] \
        == samples[(random_sample // nsamples) + nchains_train,
                    random_sample % nsamples,random_dim]


    ndim        = 5
    nsamples    = 100
    nchains     = 200
    training_proportion = 0.75

    chains_all = ch.Chains(ndim)

    np.random.seed(3)
    samples       = np.random.randn(nchains, nsamples, ndim)
    ln_posterior  = -np.sum(samples*samples, axis=2)/2.0

    chains_all.add_chains_3d(samples, ln_posterior)

    with pytest.raises(ValueError):
        chains_train, chains_test = utils.split_data(chains_all, training_proportion=-0.1)

    with pytest.raises(ValueError):
        chains_train, chains_test = utils.split_data(chains_all, training_proportion=1.5)

    with pytest.raises(ValueError):
        chains_train, chains_test = utils.split_data(chains_all, training_proportion=1E-10)

    chains_train, chains_test = utils.split_data(chains_all, training_proportion=training_proportion)


    nchains_train = int(nchains*training_proportion)

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

    nchains_test = nchains - nchains_train

    assert chains_test.nchains      == nchains_test
    assert chains_test.nsamples     == nsamples * nchains_test
    assert len(chains_test.start_indices) == nchains_test + 1
    for i in range(nchains_test + 1):
        assert chains_test.start_indices[i] == i * nsamples

    assert chains_test.samples.shape[0]        == nsamples * nchains_test
    assert chains_test.samples.shape[1]        == ndim
    assert chains_test.ln_posterior.shape[0]   == nsamples * nchains_test

    random_sample = np.random.randint(nsamples * nchains_test)
    random_dim    = 3
    assert chains_test.samples[random_sample,random_dim] \
        == samples[(random_sample // nsamples) + nchains_train,
                    random_sample % nsamples,random_dim]


def test_validation_fit_indexes():

    nchains = 10
    nfold  = 3

    np.random.seed(0)
    indexes = list(np.random.permutation(nchains)) 
        # creates [2, 8, 4, 9, 1, 6, 7, 3, 0, 5]

    nchains_in_val_set = nchains/nfold

    with pytest.raises(ValueError):
        utils.validation_fit_indexes(nfold, nchains_in_val_set, nfold, indexes)
    with pytest.raises(ValueError):
        utils.validation_fit_indexes(-1, nchains_in_val_set, nfold, indexes)
    with pytest.raises(ValueError):
        utils.validation_fit_indexes(0, nchains_in_val_set=nchains+1,
                                     nfold=nfold, indexes=indexes)
    
    indexes_val, indexes_fit = \
        utils.validation_fit_indexes(0, nchains_in_val_set, nfold, indexes)
    assert len(indexes_val) == 3
    for index_val, index_check in zip(indexes_val, [2, 8, 4]):
        assert index_val == index_check
    for index_fit, index_check in zip(indexes_fit, [9, 1, 6, 7, 3, 0, 5]):
        assert index_fit == index_check

    indexes_val, indexes_fit = \
        utils.validation_fit_indexes(1, nchains_in_val_set, nfold, indexes)
    assert len(indexes_val) == 3
    for index_val, index_check in zip(indexes_val, [9, 1, 6]):
        assert index_val == index_check
    for index_fit, index_check in zip(indexes_fit, [2, 8, 4, 7, 3, 0, 5]):
        assert index_fit == index_check

    indexes_val, indexes_fit = \
        utils.validation_fit_indexes(2, nchains_in_val_set, nfold, indexes)
    assert len(indexes_val) == 4
    for index_val, index_check in zip(indexes_val, [7, 3, 0, 5]):
        assert index_val == index_check
    for index_fit, index_check in zip(indexes_fit, [2, 8, 4, 9, 1, 6]):
        assert index_fit == index_check


def test_cross_validation():

    ndim        = 2
    nsamples    = 10
    nchains     = 200
    nfold       = 2


    hyper_parameters_HS   = [None for R in range(3)]
    hyper_parameters_KDE  = [[10**R] for R in range(-2,0)]
    hyper_parameters_MGMM = [[nguassians,1E-30,0.1*nguassians*nguassians,30,1] for nguassians in range(1,4)]

    chains = ch.Chains(ndim)

    np.random.seed(3)
    samples       = np.random.randn(nchains, nsamples, ndim)
    ln_posterior  = -np.sum(samples*samples, axis=2)/2.0

    chains.add_chains_3d(samples, ln_posterior)
    
    # just checks the result of the code is unchanged
    validation_variances = utils.cross_validation(chains, 
        [np.array([1E-1,1E1])], \
        hyper_parameters_HS, modelClass=md.HyperSphere)
    assert validation_variances[0] == pytest.approx(1.503159310628641e-05) 
    assert validation_variances[1] == pytest.approx(1.503159310628641e-05) 
    validation_variances = utils.cross_validation(chains, [], 
                                                  hyper_parameters_KDE)
    assert validation_variances[0] == pytest.approx(9.843664133455417e-05) 
    assert validation_variances[1] == pytest.approx(2.599727834904592e-06) 
    validation_variances = utils.cross_validation(chains, 
        [np.array([1E-2,10E0])], \
        hyper_parameters_MGMM, modelClass=md.ModifiedGaussianMixtureModel)
    assert validation_variances[0] == pytest.approx(1.4328193315511208e-07) 
    assert validation_variances[1] == pytest.approx(3.0406190613789445e-06, abs=2e-7)  
    assert validation_variances[2] == pytest.approx(4.572236365068644e-06, abs=2e-7) 
