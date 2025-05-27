import pytest
import numpy as np
from scipy.stats import multivariate_normal
import harmonic.model_classical as mdc
import harmonic.model as md
import harmonic.sddr as sddr
import harmonic.model_legacy as mdl

histogram_model_2D = mdc.HistogramModel(2, nbins=50)
histogram_model_1D = mdc.HistogramModel(1, nbins=50)
spline_model_1D = md.RQSplineModel(1, standardize=True, temperature=1.0)
spline_model_2D = md.RQSplineModel(2, standardize=True, temperature=1.0)
spline_model_4D = md.RQSplineModel(4, standardize=True, temperature=1.0)
nvp_model_3D = md.RealNVPModel(3, standardize=True, temperature=1.0)

model_constructors_to_test = [histogram_model_1D,
                              histogram_model_2D,
                              spline_model_2D,
                              spline_model_1D,
                              spline_model_4D,
                              nvp_model_3D]

@pytest.mark.parametrize("model", model_constructors_to_test)
def test_sddr_constructor(model):
    samples = np.random.rand(100, model.ndim)
    
    # Test valid initialization
    sddr_instance = sddr.sddr(model, samples)
    assert sddr_instance.model == model
    assert sddr_instance.samples.shape == samples.shape
    
    # Test invalid ndim
    with pytest.raises(ValueError):
        sddr.sddr(model, np.random.rand(100, 0))
    
    # Test fitted model
    model.fit(samples)
    with pytest.raises(ValueError):
        sddr.sddr(model, samples)
    
    # Test temperature
    if hasattr(model, 'temperature'):
        model.temperature = 0.5
        with pytest.raises(ValueError):
            sddr.sddr(model, samples)

def test_unsupported_model():
    samples = np.random.rand(100, 2)
    # Test unsupported model
    unsupported_model = mdl.HyperSphere(2, [np.array([1e-1, 1e1])])
    with pytest.warns(Warning):
        sddr.sddr(unsupported_model, samples)
        
def test_log_bayes_factor():
    model = md.RQSplineModel(4, standardize=True, temperature=1.0)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = np.log(multivariate_normal(mean, cov).pdf(mean))
    sddr_instance = sddr.sddr(model, training_samples)
    test_results = sddr_instance.log_bayes_factor(log_prior=0.,
                                          value=mean,
                                          nbootstraps=4,
                                          bootstrap_proportion=0.5,
                                          bootstrap=True)
    
    assert 'log_bf' in test_results
    assert 'log_bf_std' in test_results
    assert test_results['log_bf'] == pytest.approx(truth, abs=0.1 * np.abs(truth))
    assert test_results['log_bf_std'] > 0
 
def test_log_bayes_factor_kwargs():
    model = md.RealNVPModel(3, standardize=True, temperature=1.0)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = np.log(multivariate_normal(mean, cov).pdf(mean))
    sddr_instance = sddr.sddr(model, training_samples)
    results = sddr_instance.log_bayes_factor(log_prior=0.,
                                          value=mean,
                                          nbootstraps=4,
                                          bootstrap_proportion=0.5,
                                          bootstrap=True,
                                          epochs=100)
    
    assert 'log_bf' in results
    assert 'log_bf_std' in results
    assert results['log_bf'] == pytest.approx(truth, abs=0.1 * np.abs(truth))
    assert results['log_bf_std'] > 0

def test_bayes_factor():
    # Test with a single value
    model = mdc.HistogramModel(2, nbins=50)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = multivariate_normal(mean, cov).pdf(mean)
    sddr_instance = sddr.sddr(model, training_samples)
    results = sddr_instance.bayes_factor(prior=1.0,
                                         value=mean,
                                         nbootstraps=10,
                                         bootstrap_proportion=0.5,
                                         bootstrap=True)
    
    assert 'bf' in results
    assert 'bf_std' in results
    assert results['bf'] == pytest.approx(truth, abs=0.1 * np.abs(truth))
    assert results['bf_std'] > 0

def test_bayes_factor_kwargs():
    # Test with a single value
    model = md.RealNVPModel(4, standardize=True, temperature=1.0)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = multivariate_normal(mean, cov).pdf(mean)
    sddr_instance = sddr.sddr(model, training_samples)
    results = sddr_instance.bayes_factor(prior=1.0,
                                         value=mean,
                                         nbootstraps=10,
                                         bootstrap_proportion=0.5,
                                         bootstrap=True,
                                         epochs=100)
    
    assert 'bf' in results
    assert 'bf_std' in results
    assert results['bf'] == pytest.approx(truth, abs=0.1 * np.abs(truth))
    assert results['bf_std'] > 0
    
def test_log_bayes_factor_no_bootstrap():
    model = mdc.HistogramModel(2, nbins=50)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = np.log(multivariate_normal(mean, cov).pdf(mean))
    sddr_instance = sddr.sddr(model, training_samples)
    results = sddr_instance.log_bayes_factor(log_prior=0.,
                                             value=mean,
                                             bootstrap=False)
    
    assert 'log_bf' in results
    assert results['log_bf'] == pytest.approx(truth, abs=0.1 * np.abs(truth))
    assert results['log_bf_std'] is None