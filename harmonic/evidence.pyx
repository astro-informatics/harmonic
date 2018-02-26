import numpy as np
cimport numpy as np
import chains as ch
from libc.math cimport exp

class Evidence:
    """Class description here
    
    """
        
    def __init__(self, long nchains, model not None):
        """ constructor for evidence class. It sets the values
        to intial values ready for samples to be inputted

        Args:
            long nchains: the number of chains that are going to be
                used in the compuation            
            model: An instance of a posterior model class that has been fitted.
        
        Raises:
            ValueError: Raised if the number of chains is not positive.
            ValueError: Raised if the number of dimensions is not positive.            
            ValueError: Raised if model not fitted.
        """
        if nchains < 1:
            raise ValueError("nchains must be greater than 0.")

        if model.ndim < 1:
            raise ValueError("ndim must be greater than 0.")
                
        if not model.is_fitted():
            raise ValueError("Model not fitted.")

        self.running_sum = np.zeros(nchains)
        self.nsamples_per_chain = np.zeros((nchains),dtype=long)

        self.nchains = nchains
        self.ndim = model.ndim
        self.evidence_inv = 0.0
        self.evidence_inv_var = 0.0
        self.evidence_inv_var_var = 0.0

        self.mean_shift_set = False
        self.mean_shift = 0.0
        
        self.model = model

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
        """ Uses the running totals of running_sum and n_samples for
            each chain to calculates an estimate of the evidence,
            and estimate of the varience, and an estimate of the varience
            of the varience.

        Args:
            None

        Raises:
            None
        """

        cdef np.ndarray[double, ndim=1, mode="c"] running_sum = self.running_sum
        cdef np.ndarray[long, ndim=1, mode="c"] nsamples_per_chain = self.nsamples_per_chain

        cdef long i_chains, nsamples=0, nchains = self.nchains
        cdef double evidence_inv=0.0, evidence_inv_var=0.0, kur=0.0, dummy, n_eff=0

        for i_chains in range(nchains):
            evidence_inv             += running_sum[i_chains]
            nsamples += nsamples_per_chain[i_chains]
        evidence_inv /= nsamples

        for i_chains in range(nchains):
            dummy  = running_sum[i_chains]/nsamples_per_chain[i_chains]
            dummy -= evidence_inv
            n_eff += nsamples_per_chain[i_chains]*nsamples_per_chain[i_chains]
            evidence_inv_var    += nsamples_per_chain[i_chains]*dummy*dummy
            kur     += nsamples_per_chain[i_chains]*dummy*dummy*dummy*dummy

        n_eff = <double>nsamples*<double>nsamples/n_eff
        evidence_inv_var   /= nsamples
        kur    /= nsamples
        kur    /= evidence_inv_var*evidence_inv_var

        self.evidence_inv = evidence_inv*exp(-self.mean_shift)
        self.evidence_inv_var = evidence_inv_var*exp(-2*self.mean_shift)/(n_eff)
        self.evidence_inv_var_var  = evidence_inv_var**2*exp(-4*self.mean_shift)/(n_eff*n_eff*n_eff)
        self.evidence_inv_var_var *= ((kur - 1) + 2./(n_eff-1))
        return

    def add_chains(self, chains not None):
        """ Calculates an estimate of the evidence,
            and estimate of the varience, and an estimate of the varience
            of the varience. It does this using running averages of the 
            totals for each chain. This means it can be called many times
            with new samples and the evidence estimate will improve

        Args:
            chains: An instance of the chains class containing the chains
                to be used in the calculation
            

        Raises:
            ValueError: If the input number of chains to not match the number
                of chains already set up            

        Returns:
            None
        """

        if chains.nchains != self.nchains:
            raise ValueError("nchains do not match")

        if chains.ndim != self.ndim:
            raise ValueError("Chains ndim inconsistent")

        cdef np.ndarray[double, ndim=2, mode="c"] X = chains.samples
        cdef np.ndarray[double, ndim=1, mode="c"] Y = chains.ln_posterior
        cdef np.ndarray[double, ndim=1, mode="c"] running_sum = self.running_sum
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
                running_sum[i_chains] += exp( self.model.predict(X[i_samples,:]) \
                    - Y[i_samples] + mean_shift )
                nsamples_per_chain[i_chains] += 1

        self.process_run()

        return
