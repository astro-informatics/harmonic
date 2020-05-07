import numpy as np
cimport numpy as np
import chains as ch
from libc.math cimport exp
from math import fsum
import warnings
from enum import Enum
import scipy.special as sp
import pickle

# json can possibly be removed
import json


class Optimisation(Enum):
    """
    Enumeration to define whether to optimise for speed or accuracy.  In
    practices accuracy optimisation does not make a great deal of difference.
    """

    SPEED = 1
    ACCURACY = 2


OPTIMISATION = Optimisation.SPEED

class Shifting(Enum):
    """
    Enumeration to define which log-space shifting to adopt. Different choices 
    may prove optimal for certain settings.
    """
    MEAN_SHIFT = 1
    MAX_SHIFT = 2
    MIN_SHIFT = 3
    ABS_MAX_SHIFT = 4

class Evidence:
    """
    Compute inverse evidence values from chains, using posterior model.

    Multiple chains can be added in sequence (to avoid having to store very
    long chains).
    """

    def __init__(self, long nchains, model not None, \
                 shift=Shifting.MEAN_SHIFT):
        """
        Construct evidence class for computing inverse evidence values from
        set number of chains and initialised posterior model.

        Args:
            - long nchains: 
                Number of chains that will be used in the compuation.
            - model: 
                An instance of a posterior model class that has been fitted.
            - shift:
                What shifting method to use to avoid over/underflow during 
                computation. Selected from enumerate class.

        Raises:
            - ValueError: 
                Raised if the number of chains is not positive.
            - ValueError: 
                Raised if the number of dimensions is not positive.
            - ValueError: 
                Raised if model not fitted.
        """

        if nchains < 1:
            raise ValueError("nchains must be greater than 0.")

        if model.ndim < 1:
            raise ValueError("ndim must be greater than 0.")

        if not model.is_fitted():
            raise ValueError("Model not fitted.")

        self.running_sum = np.zeros(nchains)
        self.nsamples_per_chain = np.zeros((nchains),dtype=long)
        self.nsamples_eff_per_chain = np.zeros((nchains),dtype=long)

        """
        Chain parameters and realspace statistics
        """
        self.nchains = nchains
        self.ndim = model.ndim
        self.evidence_inv = 0.0
        self.evidence_inv_var = 0.0
        self.evidence_inv_var_var = 0.0
        self.kurtosis = 0.0
        self.n_eff = 0

        """
        For statistics computed purely in log-space.
        """
        self.ln_evidence_inv = 0.0
        self.ln_evidence_inv_var = 0.0
        self.ln_evidence_inv_var_var = 0.0
        self.ln_kurtosis = 0.0

        """
        Shift selection
        """
        self.shift = shift
        self.shift_set = False
        self.shift_value = 0.0

        self.chains_added = False

        self.model = model

        """
        Technical details
        """
        self.lnargmax = -np.inf
        self.lnargmin = np.inf
        self.lnprobmax = -np.inf
        self.lnprobmin = np.inf
        self.lnpredictmax = -np.inf
        self.lnpredictmin = np.inf

    def set_shift(self, double shift_value):
        """
        Set the multiplicative shift of log_e posterior values to aid
        numerical stability. 

        Args:
            - double shift_value: 
                Multiplicative shift.

        Raises:
            - ValueError: 
                Raised if shift_value is NaN .
            - ValueError:
                Raised if one attempts to set shift when another shift is 
                already set.
        """
        if not np.isfinite(shift_value):
            raise ValueError("Shift must be a number")

        if self.shift_set:
            raise ValueError("Cannot define multiple shifts!")

        self.shift_value = shift_value
        self.shift_set = True
        return

    def process_run(self):
        """
        Use the running totals of realspace running_sum and nsamples_per_chain
        to calculate an estimate of the inverse evidence, its variance,
        and the variance of the variance.

        This method is ran each time chains are added to update the inverse
        variance estimates from the running totals.

        """

        cdef np.ndarray[double, ndim=1, mode="c"] running_sum = self.running_sum
        cdef np.ndarray[long, ndim=1, mode="c"] nsamples_per_chain = \
            self.nsamples_per_chain

        cdef long i_chains, nsamples=0, nchains = self.nchains
        cdef double evidence_inv=0.0, evidence_inv_var=0.0
        cdef double kur=0.0, dummy, n_eff=0
        cdef double evidence_inv_var_exp=0.0, kur_exp=0.0

        for i_chains in range(nchains):
            if OPTIMISATION == Optimisation.SPEED:
                evidence_inv += running_sum[i_chains]
            nsamples += nsamples_per_chain[i_chains]
        if OPTIMISATION == Optimisation.ACCURACY:
            evidence_inv += fsum(running_sum)
        
        evidence_inv /= nsamples

        """
        The following code computes the exponents of the variance and variance 
        of the variance where possible in log space to avoid overflow errors.

        This is a log-space representation of the real-space statistics. One may
        alternatively compute the log-space statistics, but should note that 
        simply taking the exponential of log-space statistics is NOT the same as 
        computing the real-space statistics.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] y_i=np.zeros(len(running_sum))
        cdef np.ndarray[double, ndim=1, mode="c"] z_i=np.zeros(len(running_sum))
        cdef double y_mean=0.0, z_mean=0.0

        """
        Precompute differential vectors.
        """
        z_i[:]  = np.abs((running_sum[:]/nsamples_per_chain[:])-evidence_inv)
        y_i[:]  = z_i[:] * (nsamples_per_chain[:]**(0.25))
        z_i[:] *= nsamples_per_chain[:]**(0.5) 

        """
        Compute exponents using logsumexp for numerical stability.
        """
        evidence_inv_var_exp = sp.logsumexp(2.0*np.log(z_i)) - np.log(nsamples)
        kur_exp = sp.logsumexp(4.0 * np.log(y_i)) - np.log(nsamples) \
                                                  - 2.0 * evidence_inv_var_exp
        kur = np.exp(kur_exp)
        self.kurtosis = kur
        self.ln_kurtosis = kur_exp
        """
        Compute effective chain lengths.
        """
        for i in range(nchains):
            n_eff += nsamples_per_chain[i_chains]*nsamples_per_chain[i_chains]
        n_eff = <double>nsamples*<double>nsamples/n_eff
        self.n_eff = n_eff

        """
        Compute inverse evidence values as a log-space representation of the 
        real-space statistics to attempt to avoid float overflow.
        """
        self.ln_evidence_inv = np.log(evidence_inv) - self.shift_value
        self.ln_evidence_inv_var = evidence_inv_var_exp - 2 * self.shift_value \
                                                    - np.log(n_eff - 1)
        self.ln_evidence_inv_var_var = 2. * evidence_inv_var_exp \
                                                    - 3. * np.log(n_eff) \
                                                    - 4. * self.shift_value \
                                                    + np.log((kur - 1) + 2./(n_eff-1))

        """
        Compute inverse evidence statistics in real-space. In certain settings
        these values may return nan due to float overflow: in these cases one 
        should use the log-space values.
        """
        self.evidence_inv = exp(self.ln_evidence_inv)
        self.evidence_inv_var = exp(self.ln_evidence_inv_var)
        self.evidence_inv_var_var = exp(self.ln_evidence_inv_var_var)

        return

    def add_chains(self, chains not None):
        """
        Add new chains and calculate an estimate of the inverse evidence, its
        variance, and the variance of the variance.

        Calculations are performed by using running averages of the totals for
        each chain. Consequently, the method can be called many times with new
        samples for each chain so that the evidence estimate will improve. The
        rationale is that not all samples need to be stored in memory for
        high-dimensional problems.  Note that the same number of chains needs to
        be considered for each call.

        Args:
            - chains: 
                An instance of the chains class containing the chains to be used 
                in the calculation.

        Raises:
            - ValueError: 
                Raised if the input number of chains to not match the number of 
                chains already set up.
            - ValueError:
                Raised if both max and mean shift are set.

        """

        if chains.nchains != self.nchains:
            raise ValueError("nchains do not match")

        if chains.ndim != self.ndim:
            raise ValueError("Chains ndim inconsistent")

        cdef np.ndarray[double, ndim=2, mode="c"] X = chains.samples
        cdef np.ndarray[double, ndim=1, mode="c"] Y = chains.ln_posterior
        cdef np.ndarray[double, ndim=1, mode="c"] running_sum = self.running_sum
        cdef np.ndarray[long,   ndim=1, mode="c"] \
        nsamples_per_chain = self.nsamples_per_chain
        cdef np.ndarray[long,   ndim=1, mode="c"] \
        nsamples_eff_per_chain = self.nsamples_eff_per_chain

        cdef long i_chains, i_samples, nchains = self.nchains
        cdef double mean_shift, max_shift, max_i

        """
        The following performs a shift in log-space to avoid overflow or float
        rounding errors in realspace.
        """
        if not self.shift_set:
            if self.shift == Shifting.MAX_SHIFT:
                # Shifts by max of the log-posterior
                self.set_shift(np.max(Y))
            if self.shift == Shifting.MEAN_SHIFT:
                # Shifts by mean of the log-posterior
                self.set_shift(np.mean(Y))
            if self.shift == Shifting.MIN_SHIFT:
                # Shifts by min of the log-posterior
                self.set_shift(np.min(Y))
            if self.shift == Shifting.ABS_MAX_SHIFT:
                # Shifts by the absolute maximum of log-posterior
                self.set_shift(Y[np.argmax(np.abs(Y))])


        for i_chains in range(nchains):
            i_samples_start = chains.start_indices[i_chains]
            i_samples_end = chains.start_indices[i_chains+1]

            terms =[]  # TODO: replace terms by numpy array, set size here
            
            terms_ln =np.zeros(i_samples_end - i_samples_start)
            

            for i,i_samples in enumerate(range(i_samples_start, i_samples_end)):

                lnpredict = self.model.predict(X[i_samples,:])
                lnprob = Y[i_samples]
                lnarg = lnpredict - lnprob
                # Apply shifting term to avoid overflow.
                lnarg += self.shift_value
                # Store realspace or logspace sum depending on choice.
                term = exp(lnarg)
                nsamples_per_chain[i_chains] += 1

                if not lnpredict == -np.inf:

                    # Count number of samples used.
                    nsamples_eff_per_chain[i_chains] +=1

                    if OPTIMISATION == Optimisation.SPEED:
                        # Add contribution to running sum.
                        running_sum[i_chains] += term

                    elif OPTIMISATION == Optimisation.ACCURACY:
                        # Store all contributions in list to then be added.
                        terms.append(term)
                        terms_ln[i] = lnarg

                    # Log diagnostic terms.
                    self.lnargmax = lnarg \
                        if lnarg > self.lnargmax else self.lnargmax
                    self.lnargmin = lnarg \
                        if lnarg < self.lnargmin else self.lnargmin
                    self.lnprobmax = lnprob \
                        if lnprob > self.lnprobmax else self.lnprobmax
                    self.lnprobmin = lnprob \
                        if lnprob < self.lnprobmin else self.lnprobmin
                    self.lnpredictmax = lnpredict \
                        if lnpredict > self.lnpredictmax else self.lnpredictmax
                    self.lnpredictmin = lnpredict \
                        if lnpredict < self.lnpredictmin else self.lnpredictmin
            
            if OPTIMISATION == Optimisation.ACCURACY:
                # Sum all terms at once (fsum) to get track of partial terms.
                # Could extend this approach to compute final sums with fsum but
                # running sums for each chain should be of similar value so not
                # a great deal of merit in doing so. Hacked experiments to test
                # that (not implemented in current version) show that approach
                # doesn't make much difference.  Even doing fsum here doesn't
                # make much difference.  Recommended approach is to optimise for
                # speed rather than acuracy since accuracy optimisation doesn't
                # make much difference.
                
                # running_sum[i_chains] += fsum(terms)
                
                #running_sum[i_chains] = fsum(terms)
                
                # running_sum[i_chains] = exp(sp.logsumexp(terms_ln))
                # running_sum[i_chains] = (sp.logsumexp(terms_ln))
                # running_sum[i_chains] += sp.logsumexp(terms_ln)
                
                
                           
                if np.amax(np.absolute(terms_ln)) > np.amax(terms_ln):
                    offset = np.amin(terms_ln)
                else:
                    offset = np.amax(terms_ln)
                
                running_sum[i_chains] = np.log(np.sum(np.exp(terms_ln -offset)))

                running_sum[i_chains] += offset
                running_sum[i_chains] -= np.log(i_samples_end -i_samples_start)
                 
                # running_sum[i_chains] = np.sum(np.exp(terms_ln - offset))
                # running_sum[i_chains] *= offset

        self.process_run()
        self.chains_added = True
        self.check_basic_diagnostic()

        return

    def check_basic_diagnostic(self):
        """
        Perform basic diagontic check on sanity of evidence calulations.

        If these tests pass it does *not* necessarily mean the evidence is
        accurate and other tests should still be performed.

        Return:
            - Boolean: 
                Bool variable speciying whehter diagnostic tests pass.

        Raises:
            - Warnings: 
                Raised if the diagnostic tests fail.
        """

        NSAMPLES_EFF_WARNING_LEVEL = 30
        LNARG_WARNING_LEVEL = 10.0

        tests_pass = True

        if np.mean(self.nsamples_eff_per_chain) <= NSAMPLES_EFF_WARNING_LEVEL:
            warnings.warn("Evidence may not be accurate due to low " + \
                "number of effective samples (mean number of effective " + \
                "samples per chain is {}). Use more samples."
                .format(np.mean(self.nsamples_eff_per_chain)))
            tests_pass = False

        if (self.lnargmax - self.lnargmin) >= LNARG_WARNING_LEVEL:
            warnings.warn("Evidence may not be accurate due to large " +
                "dynamic range. Use model with smaller support " +
                "and/or better predictive accuracy.")
            tests_pass = False

        return tests_pass

    def compute_evidence(self):
        """
        Compute evidence from the inverse evidence.

        Returns: 
            - (evidence, evidence_std):
                - evidence: 
                    Estimate of evidence.
                - evidence_std: 
                    Estimate of standard deviation of evidence.
        """

        self.check_basic_diagnostic()

        common_factor = 1.0 + self.evidence_inv_var/(self.evidence_inv**2)

        evidence = common_factor / self.evidence_inv

        evidence_std = np.sqrt(self.evidence_inv_var) / (self.evidence_inv**2)

        return (evidence, evidence_std)

    def compute_ln_evidence(self):
        """
        Compute log_e of evidence from the inverse evidence.

        Returns: 
            - (ln_evidence, ln_evidence_std):
                - ln_evidence: 
                    Estimate of log_e of evidence.
                - ln_evidence_std: 
                    Estimate of log_e of standard deviation of evidence.
        """

        self.check_basic_diagnostic()

        common_factor = 1.0 + self.evidence_inv_var/(self.evidence_inv**2)

        ln_evidence = np.log(common_factor) - np.log(self.evidence_inv)

        ln_evidence_std = 0.5*np.log(self.evidence_inv_var) \
            - 2.0*np.log(self.evidence_inv)

        return (ln_evidence, ln_evidence_std)

    def serialize_evidence_class(self):
        """
        Serializes the evidence class for checkpointing.

        Returns:
            - Nothing.
        """
        # file_name = 'user.json'
        # with open(file_name, 'w') as file:
        json.dumps(self, default=convert_to_dict,indent=4, sort_keys=True)

        return

    def serialize(self, filename):
        """Serialize evidence object.

        Args:

            filename (string): Name of file to save evidence object.

        """

        file = open(filename, "wb")
        pickle.dump(self, file)
        file.close()

        return

    @classmethod
    def deserialize(self, filename):
        """Deserialize Evidence object from file.

        Args:

            filename (string): Name of file from which to read evidence object.

        Returns:

            (Evidence): Evidence object deserialized from file.

        """
        file = open(filename,"rb")
        ev = pickle.load(file)
        file.close()

        return ev





def deserialize_evidence_class(file_name):
    """
    Serializes the evidence class for checkpointing.

    Returns:
        - Nothing.
    """
    file_name = 'user.json'
    with open(file_name, 'r') as file:
        json.loads(file, object_hook=dict_to_obj)


def compute_bayes_factor(ev1, ev2):
    """
    Compute Bayes factor of two models.

    Args:
        - ev1: 
            Evidence object of model 1 with chains added.
        - ev2: 
            Evidence object of model 2 with chains added.

    Returns: 
        - (bf12, bf12_std):
            - bf12: 
                Estimate of the Bayes factor Z_1 / Z_2.
            - bf12_std: 
                Estimate of the standard deviation of the Bayes factor
                sqrt( var ( Z_1 / Z_2 ) ).

    Raises:
        - ValueError: 
            Raised if model 1 does not have chains added.
        - ValueError: 
            Raised if model 2 does not have chains added.
    """

    if not ev1.chains_added:
        raise ValueError("Evidence for model 1 does not have chains added")
    if not ev2.chains_added:
        raise ValueError("Evidence for model 2 does not have chains added")

    ev1.check_basic_diagnostic()
    ev2.check_basic_diagnostic()

    common_factor = 1.0 + ev1.evidence_inv_var/(ev1.evidence_inv**2)

    bf12 = ev2.evidence_inv / ev1.evidence_inv * common_factor

    bf12_std = np.sqrt( ev1.evidence_inv**2 * ev2.evidence_inv_var \
                        + ev2.evidence_inv**2 * ev1.evidence_inv_var ) \
                      / (ev1.evidence_inv**2)

    return (bf12, bf12_std)

def compute_ln_bayes_factor(ev1, ev2):
    """
    Computes log_e of Bayes factor of two models.

    Args:
        - ev1: 
            Evidence object of model 1 with chains added.
        - ev2: 
            Evidence object of model 2 with chains added.

    Returns: 
        - (ln_bf12, ln_bf12_std):
            - ln_bf12: 
                Estimate of log_e of the Bayes factor ln ( Z_1 / Z_2 ).
            - ln_bf12_std: 
                Estimate of log_e of the standard deviation of the Bayes
                factor ln ( sqrt( var ( Z_1 / Z_2 ) ) ).

    Raises:
        - ValueError: 
            Raised if model 1 does not have chains added.
        - ValueError: 
            Raised if model 2 does not have chains added.
    """

    if not ev1.chains_added:
        raise ValueError("Evidence for model 1 does not have chains added")
    if not ev2.chains_added:
        raise ValueError("Evidence for model 2 does not have chains added")

    ev1.check_basic_diagnostic()
    ev2.check_basic_diagnostic()

    common_factor = 1.0 + ev1.evidence_inv_var/(ev1.evidence_inv**2)

    ln_bf12 = np.log(ev2.evidence_inv) - np.log(ev1.evidence_inv) \
        + np.log(common_factor)

    factor = ev1.evidence_inv**2 * ev2.evidence_inv_var \
             + ev2.evidence_inv**2 * ev1.evidence_inv_var

    ln_bf12_std = 0.5*np.log(factor) - 2.0 * np.log(ev1.evidence_inv)

    return (ln_bf12, ln_bf12_std)



def msum(iterable):
    """
    "Full precision summation using multiple floats for intermediate values".
    
    From: http://code.activestate.com/recipes/393090/
    Rounded x+y stored in hi with the round-off stored in lo.  Together
    hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    to each partial so that the list of partial sums remains exact.
    Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps
    """

    partials = []               # sorted, non-overlapping partial sums
    for x in iterable:
        i = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                partials[i] = lo
                i += 1
            x = hi
        partials[i:] = [x]
    return sum(partials, 0.0), partials



def convert_to_dict(obj):
  """
  A function takes in a custom object and returns a dictionary representation of the object.
  This dict representation includes meta data such as the object's module and class names.
  """
  
  #  Populate the dictionary with object meta data 
  obj_dict = {
    "__class__": obj.__class__.__name__,
    "__module__": obj.__module__
  }
  
  #  Populate the dictionary with object properties
  obj_dict.update(obj.__dict__)
  
  return obj_dict

def dict_to_obj(our_dict):
    """
    Function that takes in a dict and returns a custom object associated with the dict.
    This function makes use of the "__module__" and "__class__" metadata in the dictionary
    to know which object type to create.
    """
    if "__class__" in our_dict:
        # Pop ensures we remove metadata from the dict to leave only the instance arguments
        class_name = our_dict.pop("__class__")
        
        # Get the module name from the dict and import it
        module_name = our_dict.pop("__module__")
        
        # We use the built in __import__ function since the module name is not yet known at runtime
        module = __import__(module_name)
        
        # Get the class from the module
        class_ = getattr(module,class_name)
        
        # Use dictionary unpacking to initialize the object
        obj = class_(**our_dict)
    else:
        obj = our_dict
    return obj
