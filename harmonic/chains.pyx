import numpy as np
cimport numpy as np
import copy

class Chains:
    """
    Class to store samples from multiple MCMC chains.    
    """


    def __init__(self, long ndim):   
        """Construct empty Chains for parameter space of dimension ndim.
        
        Constructor simply sets ndim. Chain samples are added by the add_chain*
        methods since we want to support setting up data for chains from
        different input data formats (e.g. data from a single chain or multiple
        chains at once).
        
        Args:

            ndim (long): Dimension of the parameter space.

        """    
        
        if ndim < 1:
            raise ValueError("ndim must be greater than 0")
        self.nchains = 0
        self.start_indices = [0] # stores start index of each chain
        self.ndim = ndim
        self.nsamples = 0
        self.samples = np.empty((0, self.ndim))
        self.ln_posterior = np.empty((0))


    def add_chain(self, np.ndarray[double,ndim=2,mode="c"] samples not None, 
                  np.ndarray[double,ndim=1,mode="c"] ln_posterior not None):
        """Add a single chain to a Chains object.
        
        Args:

            samples (double ndarray[nsamples, ndim]): Samples of a single
                chain.

            ln_posterior (double ndarray[n_new_samples]): log_e posterior
                values.
        
        Raises:

            ValueError: Raised when ndim of new chain does not match previous
                chains.

        """
                        
        nsamples_in = samples.shape[0]
        ndim_in = samples.shape[1]
        
        # Check new chain has correct ndim.
        if ndim_in != self.ndim:            
            raise ValueError("ndim of new chain does not match previous chains")
        
        if nsamples_in != ln_posterior.shape[0]:            
            raise ValueError("Length of sample and ln_posterior arrays do not " 
                + "match")
        
        self.samples = np.concatenate((self.samples, samples))
        self.ln_posterior = np.concatenate((self.ln_posterior, ln_posterior))
        self.nsamples += nsamples_in                
        self.start_indices.append(self.nsamples)
        self.nchains += 1
        
        return


    def add_chains_2d(self, np.ndarray[double,ndim=2,mode="c"] samples 
                      not None, 
                      np.ndarray[double,ndim=1,mode="c"] ln_posterior not None, 
                      long nchains_in):
        """Add a number of chains to a Chains object assuming all chains are 
        of the same length.
            
        Args:

            samples (double ndarray[nsamples_in * nchains_in, ndim]): Samples
                of multiple chains.

            ln_posterior (double ndarray[nsamples_in * nchains_in]): log_e
                posterior values. 

            long nchains_in: Number of chains to be added.
        
        Raises:

            ValueError: Raised when number of samples is not multiple of the
                number of chains.

            ValueError: Raised when ndim of new chains does not match previous
                chains.

            ValueError: Raised when posterior and samples first length are
                different.

        """

        if (samples.shape[0] % nchains_in) != 0:
            raise ValueError("The number of samples is not a multiple of the "
                + "number of chains")

        nsamples_in = samples.shape[0]
        ndim_in = samples.shape[1]

        # Check new chain has correct ndim.
        if ndim_in != self.ndim:            
            raise ValueError("ndim of new chain does not match previous chains")

        if samples.shape[0] != ln_posterior.shape[0]:            
            raise ValueError("Length of sample and ln_posterior arrays do not "
                + "match")

        cdef long i_chain, samples_per_chain = samples.shape[0] / nchains_in
        for i_chain in range(nchains_in):
            self.add_chain(
                samples[i_chain*samples_per_chain:
                        (i_chain+1)*samples_per_chain, :],
                ln_posterior[i_chain*samples_per_chain:
                             (i_chain+1)*samples_per_chain])

        return


    def add_chains_2d_list(self, np.ndarray[double,ndim=2,mode="c"] samples 
                           not None, 
                           np.ndarray[double,ndim=1,mode="c"] ln_posterior 
                           not None, 
                           long nchains_in, list chain_indexes):        
        """Add a number of chains to the chain class. Uses a list of indexes to
        determine where each chain starts and stops.
            
        Args:

            samples (double ndarray[nsamples_in * nchains_in, ndim]): Samples
                of multiple chains.

            ln_posterior (double ndarray[nsamples_in * nchains_in]): log_e
                posterior values. 

            nchains_in (long): Number of chains to be added.

            list chain_indexes (list): List of the starting index of the chains.
        
        Raises:

            ValueError: Raised when ndim of new chains does not match
                previous chains.

            ValueError: Raised when posterior and samples first length are
                different.

            ValueError: Raised when the length of the list is not nchains_in
                + 1.

        """

        nsamples_in = samples.shape[0]
        ndim_in = samples.shape[1]

        # Check new chain has correct ndim.
        if ndim_in != self.ndim:            
            raise ValueError("ndim of new chain does not match previous chains")

        if len(chain_indexes) != nchains_in+1:
            raise ValueError("Length of index list is not nchains_in + 1")

        if samples.shape[0] != ln_posterior.shape[0]:            
            raise ValueError("Length of sample and ln_posterior arrays do not "
                + "match")

        cdef long i_chain, samples_per_chain

        for i_chain in range(nchains_in):
            samples_per_chain = chain_indexes[i_chain+1] -chain_indexes[i_chain]

            self.add_chain(
                samples[i_chain*samples_per_chain:
                        (i_chain+1)*samples_per_chain, :],
                ln_posterior[i_chain*samples_per_chain:
                             (i_chain+1)*samples_per_chain])

        return


    def add_chains_3d(self, np.ndarray[double,ndim=3,mode="c"] samples 
                      not None, 
                      np.ndarray[double,ndim=2,mode="c"] ln_posterior not None):
        """Add a number of chains to a Chain object from 3D array.

        Args:

            samples(double ndarray[(nchains_in, nsamples_in, ndim]): Samples
                from multiple chains.

            ln_posterior(double ndarray[nchains_in, nsamples_in]): log_e
                posterior values.
      
        Raises:

            ValueError: Raised when ndim of new chains does not match previous
                chains.

            ValueError: Raised when posterior and samples first and second
                length are different.

        """

        nchains_in = samples.shape[0]
        nsamples_in = samples.shape[1]
        ndim_in = samples.shape[2]

        # Check new chain has correct ndim.
        if ndim_in != self.ndim:            
            raise ValueError("ndim of new chain does not match previous chains")

        if samples.shape[0] != ln_posterior.shape[0] \
            or samples.shape[1] != ln_posterior.shape[1]:            
            raise ValueError("Length of sample and ln_posterior arrays do not "
                + "match")

        cdef long i_chain
        for i_chain in range(nchains_in):
            self.add_chain(samples[i_chain,:,:], ln_posterior[i_chain,:])

        return
            

    def get_sub_chains(self, list chains_wanted):
        """Creates a new chain instance with the chains indexed in chains_wanted. 
        (Useful for cross-validation.)

        Args:

            list chains_wanted (list): List of indexes of chains that the new
                chain instance will contain.

        Returns:

            Chains: Chains object containing the chains wanted.

        Raises:

            ValueError: If any of the chains_wanted indexes are out of bounds
                i.e. outside of range 0 to nchains - 1.

        """

        new_nchains = len(chains_wanted)

        for chain_index in chains_wanted:
            if chain_index < 0 or chain_index >= self.nchains:
                raise ValueError("chains_wanted contains index out of bounds")

        sub_chains = Chains(self.ndim)

        for chain_index in chains_wanted:
            sub_chains.add_chain(\
                self.samples[self.start_indices[chain_index]:\
                             self.start_indices[chain_index+1],:],\
                self.ln_posterior[self.start_indices[chain_index]:
                                  self.start_indices[chain_index+1]])

        return sub_chains


    def get_chain_indices(self, long i):
        """Gets the start and end index of samples from a chain.

        The end index specifies the index one passed the end of the chain, i.e. 
        the chain samples can be accessed by self.samples[start:end,:].
        
        Args:

            i (long): Index of chain of which to determine start and end indices.

        Returns:

            (long, long): A tuple of the start and end index, i.e. (start, end).
            
        Raises:

            ValueError: Raised when chain number invalid.

        """
        
        if i < 0:
            raise ValueError("Chain number must be positive")
        if i >= self.nchains:
            raise ValueError("Chain number is greater than nchains-1")

        return self.start_indices[i], self.start_indices[i+1]
            

    def add(self, other):
        """Add other Chain object to this object.
        
        Args:

            other (Chains): Other Chain object to be added to this object.

        Raises:

            ValueError: Raised if the new chain has a different ndim.

        """
                
        if self.ndim != other.ndim:
            raise ValueError("ndim of other Chain object does not match this "
            + "Chain object.")
            
        if other.nsamples == 0:
            return            
        
        self.samples = np.concatenate((self.samples, other.samples))
        self.ln_posterior = np.concatenate((self.ln_posterior,
                                            other.ln_posterior))
        self.start_indices = self.start_indices \
             + list(map(lambda x : x + self.nsamples, other.start_indices[1:]))
        self.nchains += other.nchains
        self.nsamples += other.nsamples 
        
        return        


    def shallowcopy(self):
        """Performs shallow copy of the chain class (calls the module copy).

        """
        return copy.copy(self)


    def deepcopy(self):
        """Performs deep copy of the chain class (calls the module copy).

        """
        return copy.deepcopy(self)


    def nsamples_per_chain(self):
        """Compute list containing number of samples in each chain.
        
        Args:

            None.
        
        Returns:

            nsamples_per_chain (list): 1D list of length self.nchains containing the
                number of samples in each chain.

        """
        
        zipped = list(zip(self.start_indices[0:self.nchains],
                          self.start_indices[1:self.nchains+1]))
        
        nsamples_per_chain = list(map(lambda x : x[1] - x[0],  zipped))
        
        return nsamples_per_chain 


    def remove_burnin(self, nburn=100):
        """Remove burn-in samples from each chain.
        
        Args:

            nburn (int): Number of burn-in samples to remove from each chain.
        
        Raises:

            ValueError: Raised when nburn not less then number of samples in
                each chain.

        """
        
        start_indices_new = [0]
        samples_new = np.empty((0, self.ndim))
        ln_posterior_new = np.empty((0))
        
        cdef long i_sample = 0        
        cdef long i_chain, nsamples_chain
        for i_chain in range(self.nchains):
            
            start = self.start_indices[i_chain]
            end = self.start_indices[i_chain+1]
            nsamples_chain = end - start            
            
            if nburn >= nsamples_chain:
                raise ValueError("nburn must be less than " 
                    + "number of samples in chain")
                    
            samples_new = np.concatenate( \
                (samples_new, self.samples[start+nburn:end, :]))
            ln_posterior_new = np.concatenate( \
                (ln_posterior_new, self.ln_posterior[start+nburn:end]))
        
            i_sample += nsamples_chain - nburn       
                
            start_indices_new.append(i_sample)
            
        self.samples = samples_new
        self.ln_posterior = ln_posterior_new
        self.start_indices = start_indices_new
        self.nsamples = i_sample    
        
        return


    def split_into_blocks(self, nblocks=100):
        """Split chains into larger number of blocks.
        
        The intention of this method is to break chains into blocks that are
        (approximately) independent in order to get more independent chains for
        computing various statistics.
        
        Each existing chain is split into blocks (i.e. new chains),
        proportionally to the size of the current chains. Final blocks within
        each chain end up containing slightly different numbers of samples
        (since we do not ever want to throw away samples!). One could improve
        this, if required, to distribute the additional samples across all of
        the blocks of the chain.
                
        Args:

            nblocks (int): Number of new (blocked) chains to split existing chains
                into.

        Raises:

            ValueError: Returned if nblocks < the number chains

        """
        
        if nblocks <= self.nchains:
            raise ValueError("nblocks must be greater then number of chains")

        nsamples_per_chain = np.array(self.nsamples_per_chain())        
        # print("\n")
        # print("nsamples_per_chain = {}".format(nsamples_per_chain))        
        rel_size_chain = nsamples_per_chain / self.nsamples        
        # print("rel_size_chain = {}".format(rel_size_chain))        
        nblocks_per_chain = np.round(nblocks * rel_size_chain).astype(long)
        
        # Ensure no chains have zero blocks due to rounding.
        nblocks_per_chain[nblocks_per_chain == 0] = 1
        # print("nblocks_per_chain = {}".format(nblocks_per_chain))

        # Potentially adjust blocks per chain due to rounding errors.
        target_offset = nblocks - np.ndarray.sum(nblocks_per_chain)
        # print("target_offset = {}".format(target_offset))
        if target_offset != 0:            
            chain_to_adjust = np.argmax(nblocks_per_chain)
            nblocks_per_chain[chain_to_adjust] += target_offset
            if nblocks_per_chain[chain_to_adjust] < 1:
                raise ValueError("Adjusted block number for chain less than 1.")
        # print("nblocks_per_chain = {}".format(nblocks_per_chain))
        
        start_indices_new = np.array([0])
        cdef long i_chain
        for i_chain in range(self.nchains):
            start = self.start_indices[i_chain]
            end = self.start_indices[i_chain+1]
            
            step = long((end - start) // nblocks_per_chain[i_chain])
            # print("chain = {}, start = {}".format(i_chain, start))
            # print("chain = {}, end = {}".format(i_chain, end))
            # print("chain = {}, step = {}".format(i_chain, step))

            block_start_indices = start \
                + np.array(range(nblocks_per_chain[i_chain]+1), dtype=long) \
                * step
            
            block_start_indices[-1] = end
            # print("chain = {}, block_start_indices = {}"
            #.format(i_chain, block_start_indices))

            start_indices_new = np.concatenate((start_indices_new, 
                                                block_start_indices[1:]))

            # print("chain = {}, start_indices_new = {}"
            #.format(i_chain, start_indices_new))
        
        self.start_indices = start_indices_new.tolist()
        self.nchains = nblocks
        # print("nsamples_per_chain = {}".format(self.nsamples_per_chain()))
        
        return
