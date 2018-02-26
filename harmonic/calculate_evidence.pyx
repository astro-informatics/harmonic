import numpy as np
cimport numpy as np
import chains as ch
from libc.math cimport exp

class evidence():
    """Class description here
    
    """
        
    def __init__(self, long nchains, long ndim):
        """ constructor for evidence class. It sets the values
        to intial values ready for samples to be inputted

        Args:
            long nchains: the number of chains that are going to be
                used in the compuation
            long ndim: 

        Raises:
            ValueError: if the number of chains is not positive
        """
        if nchains < 1:
            raise ValueError("nchains must be greater than 0.")

        if ndim < 1:
            raise ValueError("ndim must be greater than 0.")


        self.p_i = np.zeros(nchains)
        self.nsamples_per_chain = np.zeros((nchains),dtype=int)

        self.nchains = nchains
        self.ndim = ndim
        self.p  = 0.0
        self.s2 = 0.0
        self.v2 = 0.0

        self.mean_shift_set = False
        self.mean_shift = 0.0

    def set_mean_shift(self, double mean_shift_in):
        """ Sets the multaplicative shift 
            (usually the geometric mean) of log_e posterior
            values to aid numerical stability

        Args:
            double mean_shift_in: the multaplicative shift

        Raises:
            ValueError: If mean_shift_in is a NaN 
        """
        if ~np.isfinite(mean_shift_in):
            raise ValueError("Mean shift must be a number")

        self.mean_shift = mean_shift_in
        self.mean_shift_set = True
        return

    def process_run(self):
        """ Uses the running totals of p_i and n_samples for
            each chain to calculates an estimate of the evidence,
            and estimate of the varience, and an estimate of the varience
            of the varience.

        Args:
            None

        Raises:
            None
        """

        cdef np.ndarray[double, ndim=1, mode="c"] p_i = self.p_i
        cdef np.ndarray[long, ndim=1, mode="c"] nsamples_per_chain = self.nsamples_per_chain

        cdef long i_chains, nsamples=0, nchains = self.nchains
        cdef double p=0.0, s2=0.0, k=0.0, dummy, n_eff=0

        for i_chains in range(nchains):
            p             += p_i[i_chains]
            nsamples += nsamples_per_chain[i_chains]
        p /= nsamples

        for i_chains in range(nchains):
            dummy  = p_i[i_chains]/nsamples_per_chain[i_chains]
            dummy -= p
            n_eff += nsamples_per_chain[i_chains]*nsamples_per_chain[i_chains]
            s2    += nsamples_per_chain[i_chains]*dummy*dummy
            k     += nsamples_per_chain[i_chains]*dummy*dummy*dummy*dummy

        n_eff = <double>nsamples*<double>nsamples/n_eff
        s2   /= nsamples
        k    /= nsamples
        k    /= s2*s2

        self.p   = p*exp(self.mean_shift)
        self.s2  = s2*exp(2*self.mean_shift)/(n_eff)
        self.v2  = s2**2*exp(4*self.mean_shift)/(n_eff*n_eff*n_eff)
        self.v2 *= ((k - 1) + 2./(n_eff-1))
        return

    def add_chains(self, chains not None, model not None):
        """ Calculates an estimate of the evidence,
            and estimate of the varience, and an estimate of the varience
            of the varience. It does this using running averages of the 
            totals for each chain. This means it can be called many times
            with new samples and the evidence estimate will improve

        Args:
            chains: An instance of the chains class containing the chains
                to be used in the calculation
            model: An instance of a model class that has been fitted.

        Raises:
            ValueError: If the input number of chains to not match the number
                of chains already set up
            ValueError: If the dimensions of the model and chains problems 
                do not match

        Returns:
            None
        """

        if chains.nchains != self.nchains:
            raise ValueError("nchains do not match")

        if chains.ndim != self.ndim:
            raise ValueError("Chains ndim inconsistent")

        if model.ndim != self.ndim:
            raise ValueError("Model ndim inconsistent")

        if not model.is_fitted():
            raise ValueError("Model not fitted")

        cdef np.ndarray[double, ndim=2, mode="c"] X = chains.samples
        cdef np.ndarray[double, ndim=1, mode="c"] Y = chains.ln_posterior
        cdef np.ndarray[double, ndim=1, mode="c"] p_i = self.p_i
        cdef np.ndarray[long,   ndim=1, mode="c"] nsamples_per_chain = self.nsamples_per_chain

        cdef long i_chains, i_samples, nchains = self.nchains
        cdef double mean_shift

        if ~self.mean_shift_set:
            self.set_mean_shift(np.mean(Y))
        mean_shift = self.mean_shift

        for i_chains in range(nchains):
            i_samples_start = chains.start_indices[i_chains]
            i_samples_end = chains.start_indices[i_chains+1]
            for i_samples in range(i_samples_start, i_samples_end):
                p_i[i_chains] += exp( model.predict(X[i_samples,:]) \
                    - Y[i_samples] - mean_shift )
                nsamples_per_chain[i_chains] += 1

        self.process_run()

        return
