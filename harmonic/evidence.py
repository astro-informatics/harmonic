import numpy as np
from math import fsum
from enum import Enum
import scipy.special as sp
import cloudpickle
from harmonic import logs as lg
import jax.numpy as jnp
import jax


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

    def __init__(self, nchains: int, model, shift=Shifting.MEAN_SHIFT):
        """Construct evidence class for computing inverse evidence values from
        set number of chains and initialised posterior model.

        Args:

            nchains (int): Number of chains that will be used in the
                computation.

            model (Model): An instance of a posterior model class that has
                been fitted.

            shift (Shifting): What shifting method to use to avoid over/underflow during
                computation. Selected from enumerate class.

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
        self.nsamples_per_chain = np.zeros(nchains)
        self.nsamples_eff_per_chain = np.zeros(nchains)

        # Chain parameters and realspace statistics
        self.nchains = nchains
        self.ndim = model.ndim
        self.kurtosis = 0.0
        self.n_eff = 0

        # For statistics computed purely in log-space.
        self.ln_evidence_inv = 0.0
        self.ln_evidence_inv_per_chain = None
        self.ln_evidence_inv_var = 0.0
        self.ln_evidence_inv_var_var = 0.0
        self.ln_kurtosis = 0.0

        # Shift selection
        self.shift = shift
        self.shift_set = False
        self.shift_value = 0.0

        self.chains_added = False

        self.model = model
        self.batch_calculation = hasattr(self.model, "flow")

        # Technical details
        self.lnargmax = -np.inf
        self.lnargmin = np.inf
        self.lnprobmax = -np.inf
        self.lnprobmin = np.inf
        self.lnpredictmax = -np.inf
        self.lnpredictmin = np.inf

    def set_shift(self, shift_value: float):
        """Set the shift value of log_e posterior values to aid numerical stability.

        Args:

            shift_value (float): Shift value.

        Raises:

            ValueError: Raised if shift_value is NaN.

            ValueError: Raised if one attempts to set shift when another
                shift is already set.

        """
        if not np.isfinite(shift_value):
            raise ValueError("Shift must be a number")

        if self.shift_set:
            raise ValueError("Cannot define multiple shifts!")

        self.shift_value = shift_value
        self.shift_set = True
        return

    def process_run(self):
        """Use the running totals of realspace running_sum and nsamples_per_chain to
        calculate an estimate of the inverse evidence, its variance, and the
        variance of the variance.

        This method is ran each time chains are added to update the inverse
        variance estimates from the running totals.

        """
        nsamples_per_chain = self.nsamples_per_chain

        evidence_inv = jnp.sum(self.running_sum)
        nsamples = jnp.sum(self.nsamples_per_chain)

        evidence_inv /= nsamples

        self.ln_evidence_inv_per_chain = (
            jnp.log(self.running_sum) - self.shift_value - jnp.log(nsamples_per_chain)
        )

        """
        The following code computes the exponents of the variance and variance 
        of the variance where possible in log space to avoid overflow errors.

        This is a log-space representation of the real-space statistics. One may
        alternatively compute the log-space statistics, but should note that 
        simply taking the exponential of log-space statistics is NOT the same as 
        computing the real-space statistics.
        """

        # Precompute differential vectors.
        z_i = np.abs((self.running_sum / self.nsamples_per_chain) - evidence_inv)
        y_i = z_i * (self.nsamples_per_chain ** (0.25))
        z_i *= self.nsamples_per_chain ** (0.5)

        # Compute exponents using logsumexp for numerical stability.
        evidence_inv_var_ln_temp = sp.logsumexp(2.0 * np.log(z_i)) - np.log(nsamples)
        kur_ln = (
            sp.logsumexp(4.0 * np.log(y_i))
            - np.log(nsamples)
            - 2.0 * evidence_inv_var_ln_temp
        )
        kur = np.exp(kur_ln)
        self.kurtosis = kur
        self.ln_kurtosis = kur_ln

        # Compute effective chain lengths.
        n_eff = jnp.sum(jnp.square(nsamples_per_chain))
        n_eff = nsamples * nsamples / n_eff
        self.n_eff = n_eff

        # Compute inverse evidence values as a log-space representation of the
        # real-space statistics to attempt to avoid float overflow.
        self.ln_evidence_inv = np.log(evidence_inv) - self.shift_value
        self.ln_evidence_inv_var = (
            evidence_inv_var_ln_temp - 2 * self.shift_value - np.log(n_eff - 1)
        )
        self.ln_evidence_inv_var_var = (
            2.0 * evidence_inv_var_ln_temp
            - 3.0 * np.log(n_eff)
            - 4.0 * self.shift_value
            + np.log((kur - 1) + 2.0 / (n_eff - 1))
        )

        return

    def get_masks(self, chain_start_ixs: jnp.ndarray) -> jnp.ndarray:
        """Create mask array for a 2D array of concatenated chains of different lengths.
        Args:

            chain_start_ixs (jnp.ndarray[nchains+1]): Start indices of chains
                in Chain object.

        Returns:

            jnp.ndarray[nchains,nsamples]: Mask array with each row corresponding to a chain
                and entries with boolean values depending on if given sample at that
                position is in that chain.
        """

        nsamples = chain_start_ixs[-1]
        range_vector = jnp.arange(nsamples)

        # Create a mask array by broadcasting the range vector
        masks_arr = (range_vector >= chain_start_ixs[:-1][:, None]) & (
            range_vector < chain_start_ixs[1:][:, None]
        )

        return masks_arr

    def add_chains(self, chains):
        """Add new chains and calculate an estimate of the inverse evidence, its
        variance, and the variance of the variance.

        Calculations are performed by using running averages of the totals for
        each chain. Consequently, the method can be called many times with new
        samples for each chain so that the evidence estimate will improve. The
        rationale is that not all samples need to be stored in memory for
        high-dimensional problems.  Note that the same number of chains needs to
        be considered for each call.

        Args:

            chains (Chains): An instance of the chains class containing the chains to
                be used in the calculation.

        Raises:

            ValueError: Raised if the input number of chains to not match the
                number of chains already set up.

            ValueError: Raised if both max and mean shift are set.

        """

        if chains.nchains != self.nchains:
            raise ValueError("nchains do not match")

        if chains.ndim != self.ndim:
            raise ValueError("Chains ndim inconsistent")

        X = chains.samples
        Y = chains.ln_posterior
        nchains = self.nchains

        if self.batch_calculation:
            lnpred = self.model.predict(x=X)
            lnargs = lnpred - Y
            lnargs = lnargs.at[jnp.isinf(lnargs)].set(jnp.nan)

        else:
            lnpred = np.zeros_like(Y)
            lnargs = np.zeros_like(Y)
            for i_chains in range(nchains):
                i_samples_start = chains.start_indices[i_chains]
                i_samples_end = chains.start_indices[i_chains + 1]

                for i, i_samples in enumerate(range(i_samples_start, i_samples_end)):
                    lnpredict = self.model.predict(X[i_samples, :])
                    lnpred[i_samples] = lnpredict

                    lnprob = Y[i_samples]
                    lnargs[i_samples] = lnpredict - lnprob

                    if np.isinf(lnargs[i_samples]):
                        lnargs[i_samples] = np.nan

                    if np.isinf(lnpred[i_samples]):
                        lnpred[i_samples] = np.nan

        # The following performs a shift in log-space to avoid overflow or float
        # rounding errors in realspace.
        if not self.shift_set:
            if self.shift == Shifting.MAX_SHIFT:
                # Shifts by max of the log-posterior
                self.set_shift(-np.nanmax(lnargs))
            if self.shift == Shifting.MEAN_SHIFT:
                # Shifts by mean of the log-posterior
                self.set_shift(-np.nanmean(lnargs))
            if self.shift == Shifting.MIN_SHIFT:
                # Shifts by min of the log-posterior
                self.set_shift(-np.nanmin(lnargs))
            if self.shift == Shifting.ABS_MAX_SHIFT:
                # Shifts by the absolute maximum of log-posterior
                self.set_shift(-lnargs[np.nanargmax(np.abs(lnargs))])

        def get_running_sum(lnargs, mask):
            running_sum = jnp.nansum(jnp.where(mask, jnp.exp(lnargs), 0.0))
            return running_sum

        def get_nans_per_chain(lnargs, mask):
            nans_num = jnp.sum(jnp.where(mask, jnp.isnan(lnargs), 0.0))
            return nans_num

        lnargs += self.shift_value

        masks = self.get_masks(jnp.array(chains.start_indices))

        running_sum_val = jax.vmap(get_running_sum, in_axes=(None, 0))(lnargs, masks)
        self.running_sum += running_sum_val

        # Count added number of samples per chain
        added_nsamples_per_chain = np.diff(jnp.array(chains.start_indices))
        self.nsamples_per_chain += added_nsamples_per_chain

        # Count number of NaN values per chain and subtract to get effective
        # number of added samples per chain
        nan_count_per_chain = jax.vmap(get_nans_per_chain, in_axes=(None, 0))(
            lnargs, masks
        )
        self.nsamples_eff_per_chain += added_nsamples_per_chain - nan_count_per_chain

        self.lnargmax = jnp.nanmax(lnargs)
        self.lnargmin = jnp.nanmin(lnargs)
        self.lnprobmax = jnp.nanmax(Y)
        self.lnprobmin = jnp.nanmin(Y)
        self.lnpredictmax = jnp.nanmax(lnpred)
        self.lnpredictmin = jnp.nanmin(lnpred)

        self.process_run()
        self.chains_added = True
        self.check_basic_diagnostic()

        return

    def check_basic_diagnostic(self):
        """Perform basic diagonstic check on sanity of evidence calculations.

        If these tests pass it does *not* necessarily mean the evidence is
        accurate and other tests should still be performed.

        Return:

            Boolean: Whether diagnostic tests pass.

        Raises:

            Warnings: Raised if the diagnostic tests fail.

        """

        NSAMPLES_EFF_WARNING_LEVEL = 30
        LNARG_WARNING_LEVEL = 1000.0

        tests_pass = True

        if np.mean(self.nsamples_eff_per_chain) <= NSAMPLES_EFF_WARNING_LEVEL:
            lg.warning_log(
                "Evidence may not be accurate due to low "
                + "number of effective samples (mean number of effective "
                + "samples per chain is {}). Use more samples.".format(
                    np.mean(self.nsamples_eff_per_chain)
                )
            )
            tests_pass = False

        if (self.lnargmax - self.lnargmin) >= LNARG_WARNING_LEVEL:
            lg.warning_log(
                "Evidence may not be accurate due to large "
                + "dynamic range. Use model with smaller support "
                + "and/or better predictive accuracy."
            )
            tests_pass = False

        return tests_pass

    def compute_evidence(self):
        """Compute evidence from the inverse evidence.

        Returns:

            (double, double): Tuple containing the following.

                - evidence (double): Estimate of evidence.

                - evidence_std (double): Estimate of standard deviation of
                  evidence.

        Raises:

            ValueError: if inverse evidence or its variance overflows.
        """

        self.check_basic_diagnostic()

        evidence_inv = np.exp(self.ln_evidence_inv)
        evidence_inv_var = np.exp(self.ln_evidence_inv_var)

        if np.isinf(np.nan_to_num(evidence_inv, nan=np.inf)) or np.isinf(
            np.nan_to_num(evidence_inv_var, nan=np.inf)
        ):
            raise ValueError(
                "Evidence is too large to represent in non-log space. Use log-space values instead."
            )

        common_factor = 1.0 + evidence_inv_var / (evidence_inv**2)

        evidence = common_factor / evidence_inv

        evidence_std = np.sqrt(evidence_inv_var) / (evidence_inv**2)

        return (evidence, evidence_std)

    def compute_ln_evidence(self):
        """Compute log_e of evidence from the inverse evidence.

        Returns:

            (double, double): Tuple containing the following.

                - ln_evidence (double): Estimate of log_e of evidence.

                - ln_evidence_std (double): Estimate of log_e of standard
                    deviation of evidence.

        """

        self.check_basic_diagnostic()

        ln_x = self.ln_evidence_inv_var - 2.0 * self.ln_evidence_inv
        x = np.exp(ln_x)
        ln_evidence = np.log(1.0 + x) - self.ln_evidence_inv
        ln_evidence_std = 0.5 * self.ln_evidence_inv_var - 2.0 * self.ln_evidence_inv

        return (ln_evidence, ln_evidence_std)

    def compute_ln_inv_evidence_errors(self):
        r"""Compute lower and uppper errors on the log_e of the inverse evidence.

        Compute the log-space error :math:`\hat{\zeta}_\pm` defined by

        .. math::

            \log ( \hat{\rho} \pm \hat{\sigma} ) = \log (\hat{\rho}) + \hat{\zeta}_\pm .

        Computed in a numerically stable way by

        .. math::

            \hat{\zeta}_\pm = \log(1 \pm \hat{\sigma} / \hat{\rho}) .

        Returns:

            (double, double): Tuple containing the following.

                - ln_evidence_err_neg (double): Lower error for log_e of inverse evidence.

                - ln_evidence_err_pos (double): Upper error for log_e of inverse evidence.

        """

        ln_ratio = 0.5 * self.ln_evidence_inv_var - self.ln_evidence_inv

        ratio = np.exp(ln_ratio)

        if np.abs(ratio - 1.0) > 1e-8:
            ln_evidence_err_neg = np.log(1.0 - ratio)
        else:
            ln_evidence_err_neg = np.NINF

        ln_evidence_err_pos = np.log(1.0 + ratio)

        return (ln_evidence_err_neg, ln_evidence_err_pos)

    def serialize(self, filename):
        """Serialize evidence object.

        Args:

            filename (string): Name of file to save evidence object.

        """

        file = open(filename, "wb")
        cloudpickle.dump(self, file)
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
        file = open(filename, "rb")
        ev = cloudpickle.load(file)
        file.close()

        return ev


def compute_bayes_factor(ev1, ev2):
    """Compute Bayes factor of two models.

    Args:

        ev1 (float): Evidence value of model 1 with chains added.

        ev2 (float): Evidence value of model 2 with chains added.

    Returns:

        (float, float): Tuple containing the following.

            - bf12: Estimate of the Bayes factor Z_1 / Z_2.

            - bf12_std: Estimate of the standard deviation of the Bayes factor
                sqrt( var ( Z_1 / Z_2 ) ).

    Raises:

        ValueError: Raised if model 1 does not have chains added.

        ValueError: Raised if model 2 does not have chains added.

        ValueError: If inverse evidence or its variance for model 1 or model 2 too large
            to store in non-log space.

    """

    if not ev1.chains_added:
        raise ValueError("Evidence for model 1 does not have chains added")
    if not ev2.chains_added:
        raise ValueError("Evidence for model 2 does not have chains added")

    ev1.check_basic_diagnostic()
    ev2.check_basic_diagnostic()

    evidence_inv_ev1 = np.exp(ev1.ln_evidence_inv)
    evidence_inv_var_ev1 = np.exp(ev1.ln_evidence_inv_var)

    evidence_inv_ev2 = np.exp(ev2.ln_evidence_inv)
    evidence_inv_var_ev2 = np.exp(ev2.ln_evidence_inv_var)

    if np.isinf(np.nan_to_num(evidence_inv_ev1, nan=np.inf)) or np.isinf(
        np.nan_to_num(evidence_inv_var_ev1, nan=np.inf)
    ):
        raise ValueError(
            "Evidence for model 1 is too large to represent in non-log space. Use log-space values instead."
        )
    if np.isinf(np.nan_to_num(evidence_inv_ev2, nan=np.inf)) or np.isinf(
        np.nan_to_num(evidence_inv_var_ev2, nan=np.inf)
    ):
        raise ValueError(
            "Evidence for model 2 is too large to represent in non-log space. Use log-space values instead."
        )

    common_factor = 1.0 + evidence_inv_var_ev1 / (evidence_inv_ev1**2)

    bf12 = evidence_inv_ev2 / evidence_inv_ev1 * common_factor

    bf12_std = np.sqrt(
        evidence_inv_ev1**2 * evidence_inv_var_ev2
        + evidence_inv_ev2**2 * evidence_inv_var_ev1
    ) / (evidence_inv_ev1**2)

    return (bf12, bf12_std)


def compute_ln_bayes_factor(ev1, ev2):
    """Computes log_e of Bayes factor of two models.

    Args:

        ev1 (float): Evidence object of model 1 with chains added.

        ev2 (float): Evidence object of model 2 with chains added.

    Returns:

        (float, float): Tuple containing the following.

            - ln_bf12: Estimate of log_e of the Bayes factor ln ( Z_1 / Z_2 ).

            - ln_bf12_std: Estimate of log_e of the standard deviation of the
                Bayes factor ln ( sqrt( var ( Z_1 / Z_2 ) ) ).

    Raises:

        ValueError: Raised if model 1 does not have chains added.

        ValueError: Raised if model 2 does not have chains added.

    """

    if not ev1.chains_added:
        raise ValueError("Evidence for model 1 does not have chains added")
    if not ev2.chains_added:
        raise ValueError("Evidence for model 2 does not have chains added")

    ev1.check_basic_diagnostic()
    ev2.check_basic_diagnostic()

    evidence_inv_ev1 = np.exp(ev1.ln_evidence_inv)
    evidence_inv_var_ev1 = np.exp(ev1.ln_evidence_inv_var)

    evidence_inv_ev2 = np.exp(ev2.ln_evidence_inv)
    evidence_inv_var_ev2 = np.exp(ev2.ln_evidence_inv_var)

    if np.isnan(evidence_inv_ev1) or np.isnan(evidence_inv_var_ev1):
        raise ValueError(
            "Evidence for model 1 is too large to represent in non-log space. Use log-space values instead."
        )
    if np.isnan(evidence_inv_ev2) or np.isnan(evidence_inv_var_ev2):
        raise ValueError(
            "Evidence for model 2 is too large to represent in non-log space. Use log-space values instead."
        )

    common_factor = 1.0 + evidence_inv_var_ev1 / (evidence_inv_ev1**2)

    ln_bf12 = (
        np.log(evidence_inv_ev2) - np.log(evidence_inv_ev1) + np.log(common_factor)
    )

    factor = (
        evidence_inv_ev1**2 * evidence_inv_var_ev2
        + evidence_inv_ev2**2 * evidence_inv_var_ev1
    )

    ln_bf12_std = 0.5 * np.log(factor) - 2.0 * np.log(evidence_inv_ev1)

    return (ln_bf12, ln_bf12_std)
