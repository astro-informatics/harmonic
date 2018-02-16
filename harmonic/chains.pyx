import numpy as np
cimport numpy as np
import copy

class Chains:
    """Class to store samples from multiple MCMC chains.    
    """
    
    
    def __init__(self, int ndim):   
        """Construct empty Chains.
        
        Constructor doesn't do anything since we want to support setting up data from different input data formats (e.g. data from a single chain or multiple chains at once).  Data is added by the add_chain* methods.        
        """    
        
        if ndim < 1:
            raise ValueError("ndim must be greater than 0")
        self.nchains = 0        
        self.start_indices = [0] 
        self.ndim = ndim
        self.nsamples = 0        
        self.samples = np.empty((0, self.ndim))
        
        
        
        
    def add_chain(self, np.ndarray[double,ndim=2,mode="c"] samples not None):
        """
        
        Args:
            samples: 2D numpy.ndarray containing the samples with shape (n_new_samples,ndim) and dtype double
        
        
        Raises:
            TypeError: Raised when ndim of new chain does not match previous chains.
        """
                        
        nsamples_new = samples.shape[0]
        ndim_new     = samples.shape[1]
        
        # Check new chain has correct ndim.
        if ndim_new != self.ndim:            
            raise TypeError("ndim of new chain does not match previous chains")
        
        self.samples = np.concatenate((self.samples, samples))
        self.nsamples += nsamples_new                
        self.start_indices.append(self.nsamples)
        self.nchains += 1
        
        
    def add_chains_2d(self, np.ndarray[double,ndim=2,mode="c"] samples not None, int n_chains_in):
        """ Adds a number of chains to the chain class assumes all the chains are of the 
            same length

        Args:
            samples: 2D numpy.ndarray containing the samples with shape (n_new_samples*n_chains_in,ndim) and dtype double
            n_chains_in: int specifying the number of chains.
        
        """

        if (samples.shape[0] % n_chains_in) != 0:
            raise ValueError("The number of samples is not a multiple of the nunber of chains")

        # nsamples_new = samples.shape[0]
        # ndim_new     = samples.shape[1]

        # Check new chain has correct ndim.
        if samples.shape[1] != self.ndim:            
            raise TypeError("ndim of new chain does not match previous chains")

        cdef int i_chain, samples_per_chain = samples.shape[0]/n_chains_in
        for i_chain in range(n_chains_in):
            self.add_chain(samples[i_chain*samples_per_chain:(i_chain+1)*samples_per_chain,:])

        return

    def add_chains_3d(self, np.ndarray[double,ndim=3,mode="c"] samples not None):
        """ Adds a number of chains to the chain class assumes all the chains from 3D array

        Args:
            samples: 2D numpy.ndarray containing the samples with shape (n_chains_in,n_new_samples,ndim) and dtype double
        
        """

        # Check new chain has correct ndim.
        if samples.shape[2] != self.ndim:            
            raise TypeError("ndim of new chain does not match previous chains")

        cdef int i_chain
        for i_chain in range(samples.shape[0]):
            self.add_chain(samples[i_chain,:,:])

        return
            
    def get_chain_idexes(self, int i):
        """ Gets the start and index of samples from a chain
        Args:
            i: The chain that you want to know the start index of

        Returns:
            A tuple of the start and end index
        """
        if i < 0:
            raise ValueError("Chain number must be positive")
        if i >= self.nchains:
            raise ValueError("Chain number is greater than n_chains-1")

        return self.start_indices[i], self.start_indices[i+1]
                                                        
    def copy(self):
        return copy.copy(self)

