import numpy as np
cimport numpy as np

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
            samples: n_new_samples x ndim
        
        
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
        
        
    def add_chains(self, samples):
        pass
        
        
    def add_chains_3d(self, samples):
        pass
        
        
    
    
    
    def get_chain(i):
        pass
        # check i valid
        # return self.samples[self.start_indices[i]:
        #                     self.start_indices[i+1]]                            
                            
    def get_nsamples_in_chain(i): 
        pass   
        # check i valid        
        # return (self.start_indices[i+1] - self.start_indices[i])
                            