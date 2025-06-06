import pytest
import harmonic.model_classical as mdc
import numpy as np
from scipy.stats import multivariate_normal

histogram_model_2D = mdc.HistogramModel(2, nbins=20)
histogram_model_1D = mdc.HistogramModel(1, nbins=20)

model_classes = [mdc.HistogramModel]
models_to_test = [histogram_model_2D, histogram_model_1D]

def test_histogram_model_constructor():
    with pytest.raises(ValueError):
        mdc.HistogramModel(0)
    with pytest.raises(ValueError):
        mdc.HistogramModel(-1)
    with pytest.raises(ValueError):
        mdc.HistogramModel(2,-1)
        
    ndim = 4
    nbins = 50
    
    hist = mdc.HistogramModel(ndim, nbins)
    
    assert hist.ndim == ndim
    assert hist.nbins == nbins

@pytest.mark.parametrize("model", models_to_test)
def test_histogram_is_fitted(model):
    ndim = model.ndim
    assert model.is_fitted() is False
    training_samples = np.random.rand(1000, ndim)
    model.fit(training_samples)
    assert model.is_fitted() is True
    
def test_histogram_fit():
    model = mdc.HistogramModel(2, nbins=20)
    with pytest.raises(ValueError):
        model.fit(np.random.rand(1000, 3))
        
def test_histogram_out_of_range_predict():
    model = mdc.HistogramModel(2, nbins=20)
    training_samples = np.random.rand(1000, 2)
    model.fit(training_samples)
    
    with pytest.raises(ValueError):
        model.predict(np.array([-1.0]))
        
    with pytest.raises(ValueError):
        model.predict(np.array([2.0]))

@pytest.mark.parametrize("model", models_to_test)
def test_histogram_predict(model):
    np.random.seed(42)
    ndim = model.ndim 
    mean = np.zeros(ndim)
    cov = np.eye(ndim)
    training_samples = np.random.multivariate_normal(mean, cov, size=10000)
    model.fit(training_samples)
    predict_prob = np.exp(model.predict(mean))
    true_prob = multivariate_normal(mean, cov).pdf(mean)
    assert predict_prob == pytest.approx(true_prob, abs=0.1*true_prob), (
            f"Prediction probability of {predict_prob:.4f} does not match true probability of {true_prob:.4f}."
        ) 