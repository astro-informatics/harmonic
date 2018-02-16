import numpy as np
cimport numpy as np
import copy

class Chains:
    """Class to store samples from multiple MCMC chains.    
    """
        
    def __init__(self, int ndim):   
        """Construct empty Chains for parameter space of dimension ndim.
        
        Constructor simply sets ndim.  Chain sampeles are added by the
        add_chain* methods since we want to support setting up data for chains
        from different input data formats (e.g. data from a single chain or
        multiple chains at once).
        
        Args:
            ndim: Dimension of the parameter space.
        """    
        
        if ndim < 1:
            raise ValueError("ndim must be greater than 0")
        self.nchains = 0        
        self.start_indices = [0]  # stores start index of each chain
        self.ndim = ndim
        self.nsamples = 0        
        self.samples = np.empty((0, self.ndim))
        
    def add_chain(self, np.ndarray[double,ndim=2,mode="c"] samples not None):
        """Add a single chain.
        
        Args:
            samples: 2D numpy.ndarray containing the samples of a single chain 
                with shape (nsamples_in, ndim_in) and dtype double.
                
        Raises:
            TypeError: Raised when ndim of new chain does not match previous 
                chains.
        """
                        
        nsamples_in = samples.shape[0]
        ndim_in = samples.shape[1]
        
        # Check new chain has correct ndim.
        if ndim_in != self.ndim:            
            raise TypeError("ndim of new chain does not match previous chains")
        
        self.samples = np.concatenate((self.samples, samples))
        self.nsamples += nsamples_in                
        self.start_indices.append(self.nsamples)
        self.nchains += 1
        
        return
        
    def add_chains_2d(self, np.ndarray[double,ndim=2,mode="c"] samples not None,
                      int nchains_in):                
        """Add multiple chains stored in a concatenated numpy.ndarray, assuming
        all the chains are of the same length.

        Args:
            samples: 2D numpy.ndarray containing the samples with shape 
                (nsamples_in * nchains_in, ndim) and dtype double.
            nchains_in: int specifying the number of chains.
        
        Raises:
            ValueError: Raised when number of samples is not multiple of the   
                number of chains.
            TypeError: Raised when ndim of new chains does not match previous 
                chains.
        """

        if (samples.shape[0] % nchains_in) != 0:
            raise ValueError("The number of samples is not a multiple of the "
                + "number of chains")

        nsamples_in = samples.shape[0]
        ndim_in = samples.shape[1]

        # Check new chain has correct ndim.
        if ndim_in != self.ndim:            
            raise TypeError("ndim of new chain does not match previous chains")

        cdef int i_chain, samples_per_chain = nsamples_in/nchains_in
        for i_chain in range(nchains_in):
            self.add_chain(samples[i_chain*samples_per_chain:
                                   (i_chain+1)*samples_per_chain,:])

        return

    def add_chains_3d(self, 
                      np.ndarray[double,ndim=3,mode="c"] samples not None):
        """Add multiple chains stored in a 3D array, assuming all the chains 
        are of the same length.

        Args:
            samples: 3D numpy.ndarray containing the samples with shape 
                (nchains_in, nsamples_in, ndim) and dtype double.
        
        Raises: 
            TypeError: Raised when ndim of new chains does not match previous 
                chains.
        """

        nchains_in = samples.shape[0]
        nsamples_in = samples.shape[1]
        ndim_in = samples.shape[2]

        # Check new chain has correct ndim.
        if ndim_in != self.ndim:            
            raise TypeError("ndim of new chain does not match previous chains")

        cdef int i_chain
        for i_chain in range(nchains_in):
            self.add_chain(samples[i_chain,:,:])

        return
            
    def get_chain_idexes(self, int i):
        """Gets the start and end index of samples from a chain.
        
        The end index specifies the index one passed the end of the chain, i.e. 
        the chain samples can be accessed by self.samples[start:end,:].
        
        Args:
            i: Index of chain of which to determine start and end indexes.

        Returns:
            A tuple of the start and end index, i.e. (start, end).
            
        Raises:
            ValueError: Raised when chain number invalid.
        """
        
        if i < 0:
            raise ValueError("Chain number must be positive")
        if i >= self.nchains:
            raise ValueError("Chain number is greater than nchains-1")

        return self.start_indices[i], self.start_indices[i+1]
                                                        
    def copy(self):
        """Copy chain.
        """
        
        return copy.copy(self)

