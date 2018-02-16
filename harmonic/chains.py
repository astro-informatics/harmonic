
class Chains:
    """Doc string here"""
    
    def __init__(self, samples=empty ndarray):
        self.nchains = 1        
        self.ndim, self.nsamples = samples.shape
        self.samples = samples
        self.chain_start_indices = [0] #ndarray?
        self.chain_start_indices.append(self.nsamples)
        
    def add_chain(self, samples):
        # check dimension correct
        self.nchains += 1
        self.samples.append(samples)
        self.nsamples += samples.shape[0] # right dimension?
        self.chain_start_indices.append(self.nsamples)
    
    def get_chain(i):
        # check i valid
        return self.samples[self.chain_start_indices[i]:
                            self.chain_start_indices[i+1]]                            
                            
    def get_nsamples_in_chain(i):    
        # check i valid        
        return (self.chain_start_indices[i+1] - self.chain_start_indices[i])
                            