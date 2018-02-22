import calculate_evidence as cbe
import pytest
import numpy as np
from scipy.stats import kurtosis
import chains as ch
import model as md

def test_constructor():

	with pytest.raises(ValueError):
		rho = cbe.evidence(0)

	nchains = 100

	rho = cbe.evidence(nchains)

	assert rho.nchains        == nchains
	assert rho.p              == pytest.approx(0.0)
	assert rho.s2             == pytest.approx(0.0)
	assert rho.v2             == pytest.approx(0.0)
	assert rho.p_i.size       == nchains 
	assert rho.n_samples.size == nchains 
	assert rho.mean_shift     == pytest.approx(0.0)
	assert rho.mean_shift_set == False
	for i_chain in range(nchains):
		assert rho.p_i[i_chain]       == pytest.approx(0.0)
		assert rho.n_samples[i_chain] == 0

def test_set_mean_shift():
	nchains = 100
	rho = cbe.evidence(nchains)

	with pytest.raises(ValueError):
		rho.set_mean_shift(np.nan)

	rho.set_mean_shift(2.0)

	assert rho.mean_shift      == pytest.approx(2.0)
	assert rho.mean_shift_set  == True

def test_end_run():

	nchains = 10
	n_samples = 20

	rho = cbe.evidence(nchains)

	np.random.seed(1)
	samples   = np.random.randn(nchains,n_samples)
	rho.p_i       = np.sum(samples,axis=1)
	rho.n_samples = np.ones(nchains, dtype=int)*n_samples
	rho.end_run()

	p  = np.mean(samples)
	s2 = np.std(np.sum(samples,axis=1)/n_samples)**2/(nchains)
	print(np.std(np.sum(samples,axis=1)/n_samples)**2, nchains)
	v2 = s2**2*(kurtosis(np.sum(samples,axis=1)/n_samples) + 2 + 2/(nchains-1))/nchains

	assert rho.p  == pytest.approx(p,abs=1E-7)
	assert rho.s2 == pytest.approx(s2)
	assert rho.v2 == pytest.approx(v2)

	np.random.seed(1)
	mean_shift     = 1.0
	samples_scaled = np.random.randn(nchains,n_samples)*np.exp(-mean_shift)
	samples        = samples_scaled*np.exp(mean_shift)
	rho.p_i        = np.sum(samples_scaled,axis=1)
	rho.n_samples  = np.ones(nchains, dtype=int)*n_samples
	rho.mean_shift = mean_shift
	rho.end_run()

	p  = np.mean(samples)
	s2 = np.std(np.sum(samples,axis=1)/n_samples)**2/(nchains)
	v2 = s2**2*(kurtosis(np.sum(samples,axis=1)/n_samples) + 2 + 2./(nchains-1))/nchains

	assert rho.p  == pytest.approx(p,abs=1E-7)
	assert rho.s2 == pytest.approx(s2)
	assert rho.v2 == pytest.approx(v2)

def test_calculate_evidence():

	nchains   = 200
	nsamples  = 500
	ndim      = 2

	# create classes
	domain = [np.array([1E-1,1E1])]
	sphere = md.HyperSphere(ndim, domain)
	chain  = ch.Chains(ndim)
	cal_ev = cbe.evidence(nchains)

	# create samples of unnormalised Gaussian
	np.random.seed(30)
	X = np.random.randn(nchains,nsamples,ndim)
	Y = -np.sum(X*X,axis=2)/2.0

	# Add samples to chains
	chain.add_chains_3d(X, Y)
	# Fit the Hyper_sphere
	sphere.fit(chain.samples,chain.ln_posterior)

	print(sphere.centres, sphere.inv_covarience, sphere.R)

	sphere_dum = md.HyperSphere(ndim+1, domain)
	with pytest.raises(ValueError):
		cal_ev.calculate_evidence(chain,sphere_dum)

	# Calculate evidence
	cal_ev.calculate_evidence(chain,sphere)

	print(cal_ev.p_i[0:10])
	print(cal_ev.mean_shift)


	assert cal_ev.p  == pytest.approx(10.158667416) 
	assert cal_ev.s2 == pytest.approx(1.271069471e-06)
	assert cal_ev.v2 == pytest.approx(1.171406675e-14)

	return

