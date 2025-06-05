import pytest
import numpy as np
from scipy.stats import multivariate_normal, norm
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
     
def test_sddr_temperature():
    samples = np.random.rand(100, 2)
    model = md.RQSplineModel(2, standardize=True, temperature=0.8)
    # Test valid temperature
    with pytest.raises(ValueError):
        sddr.sddr(model, samples)

def test_unsupported_model():
    samples = np.random.rand(100, 2)
    # Test unsupported model
    unsupported_model = mdl.HyperSphere(2, [np.array([1e-1, 1e1])])
    with pytest.warns(Warning):
        sddr.sddr(unsupported_model, samples)

def test_1D_nvp_model_error():
    samples = np.random.rand(100, 1)
    nvp_model_1D = md.RealNVPModel(1, standardize=True, temperature=1.0)
    # Test RealNVPModel in 1D
    with pytest.raises(ValueError):
        sddr.sddr(nvp_model_1D, samples)
        
def test_log_bayes_factor_1D():
    model = md.RQSplineModel(1, standardize=True, temperature=1.0)
    mean = 0.
    std = 1.0
    training_samples = np.random.normal(mean, std, size=10000)
    truth = np.log(norm(mean, std).pdf(mean))
    sddr_instance = sddr.sddr(model, training_samples)
    
    log_bf, log_bf_std = sddr_instance.log_bayes_factor(log_prior=0.,
                                        value=mean,
                                        nbootstraps=4,
                                        bootstrap_proportion=0.5,
                                        bootstrap=True)
    
    assert log_bf == pytest.approx(truth, abs=0.2 * np.abs(truth))
    assert log_bf_std > 0
        
def test_log_bayes_factor():
    model = md.RQSplineModel(4, standardize=True, temperature=1.0)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = np.log(multivariate_normal(mean, cov).pdf(mean))
    sddr_instance = sddr.sddr(model, training_samples)
    
    log_bf, log_bf_std = sddr_instance.log_bayes_factor(log_prior=0.,
                                          value=mean,
                                          nbootstraps=4,
                                          bootstrap_proportion=0.5,
                                          bootstrap=True)
    
    assert log_bf == pytest.approx(truth, abs=0.2 * np.abs(truth))
    assert log_bf_std > 0
 
def test_log_bayes_factor_kwargs():
    model = md.RealNVPModel(3, standardize=True, temperature=1.0)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = np.log(multivariate_normal(mean, cov).pdf(mean))
    sddr_instance = sddr.sddr(model, training_samples)
    log_bf, log_bf_std = sddr_instance.log_bayes_factor(log_prior=0.,
                                          value=mean,
                                          bootstrap=True,
                                          nbootstraps=4,
                                          bootstrap_proportion=0.5,
                                          epochs=100)
    
    assert log_bf == pytest.approx(truth, abs=0.2 * np.abs(truth))
    assert log_bf_std > 0

def test_bayes_factor():
    # Test with a single value
    model = mdc.RQSplineModel(2, standardize=True, temperature=1.0)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = multivariate_normal(mean, cov).pdf(mean)
    sddr_instance = sddr.sddr(model, training_samples)
    bf, bf_std = sddr_instance.bayes_factor(prior=1.0,
                                         value=mean,
                                         bootstrap=True,
                                         nbootstraps=100,
                                         bootstrap_proportion=0.5)
    
    assert bf == pytest.approx(truth, abs=0.2 * np.abs(truth))
    assert bf_std > 0

def test_bayes_factor_kwargs():
    # Test with a single value
    model = md.RQSplineModel(3, standardize=True, temperature=1.0)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = multivariate_normal(mean, cov).pdf(mean)
    sddr_instance = sddr.sddr(model, training_samples)
    bf, bf_std = sddr_instance.bayes_factor(prior=1.0,
                                         value=mean,
                                         bootstrap=True,
                                         nbootstraps=10,
                                         bootstrap_proportion=0.5,
                                         epochs=10)
    
    assert bf == pytest.approx(truth, abs=0.2 * np.abs(truth))
    assert bf_std > 0
    
def test_log_bayes_factor_no_bootstrap():
    model = md.RQSplineModel(2, standardize=True, temperature=1.0)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    truth = np.log(multivariate_normal(mean, cov).pdf(mean))
    sddr_instance = sddr.sddr(model, training_samples)
    log_bf, log_bf_std = sddr_instance.log_bayes_factor(log_prior=0.,
                                             value=mean,
                                             bootstrap=False)
    
    assert log_bf == pytest.approx(truth, abs=0.2 * np.abs(truth))
    assert log_bf_std is None