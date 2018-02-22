import numpy as np
cimport numpy as np
import chains as ch
from libc.math cimport exp

class evidence():
	def __init__(self, int nchains):

		if nchains < 1:
			raise ValueError("nchains must be greater then 0")

		self.p_i       = np.zeros(nchains)
		self.n_samples = np.zeros((nchains),dtype=int)

		self.nchains = nchains
		self.p  = 0.0
		self.s2 = 0.0
		self.v2 = 0.0

		self.mean_shift_set = False
		self.mean_shift     = 0.0

	def set_mean_shift(self, double mean_shift_in):
		if ~np.isfinite(mean_shift_in):
			raise ValueError("Mean shift must be a number")

		self.mean_shift     = mean_shift_in
		self.mean_shift_set = True
		return

	def end_run(self):

		cdef np.ndarray[double, ndim=1, mode="c"] p_i       = self.p_i
		cdef np.ndarray[long, ndim=1, mode="c"]   n_samples = self.n_samples

		cdef long i_chains, n_samples_tot=0, nchains = self.nchains
		cdef double p=0.0, s2=0.0, k=0.0, dummy, n_eff=0

		for i_chains in range(nchains):
			p             += p_i[i_chains]
			n_samples_tot += n_samples[i_chains]
		p /= n_samples_tot

		for i_chains in range(nchains):
			dummy  = p_i[i_chains]/n_samples[i_chains]
			dummy -= p
			n_eff += n_samples[i_chains]*n_samples[i_chains]
			s2    += n_samples[i_chains]*dummy*dummy
			k     += n_samples[i_chains]*dummy*dummy*dummy*dummy

		n_eff = <double>n_samples_tot*<double>n_samples_tot/n_eff
		s2   /= n_samples_tot
		k    /= n_samples_tot
		k    /= s2*s2

		self.p   = p*exp(self.mean_shift)
		self.s2  = s2*exp(2*self.mean_shift)/(n_eff)
		self.v2  = s2**2*exp(4*self.mean_shift)/(n_eff*n_eff*n_eff)
		self.v2 *= ((k - 1) + 2./(n_eff-1))
		return

	def calculate_evidence(self, chain not None, model not None):

		if chain.nchains != self.nchains:
			raise ValueError("nchains do not match")

		if chain.ndim != model.ndim:
			raise ValueError("Model and chains ndim's do not match")

		cdef np.ndarray[double, ndim=2, mode="c"] X=chain.samples
		cdef np.ndarray[double, ndim=1, mode="c"] Y=chain.ln_posterior, p_i = self.p_i
		cdef np.ndarray[long,   ndim=1, mode="c"] n_samples = self.n_samples

		cdef long i_chains, i_samples, nchains = self.nchains
		cdef double mean_shift

		if ~self.mean_shift_set:
			self.set_mean_shift(np.mean(Y))
		mean_shift = self.mean_shift

		for i_chains in range(nchains):
			for i_samples in range(chain.start_indices[i_chains],chain.start_indices[i_chains+1]):
				p_i[i_chains]        += exp(model.predict(X[i_samples,:]) - Y[i_samples] - mean_shift)
				n_samples[i_chains]  += 1

		self.end_run()

		return
