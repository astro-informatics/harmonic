import pytest
import numpy as np
from scipy.stats import kurtosis
import harmonic.chains as ch
import harmonic.model_legacy as mdl
import harmonic.evidence as cbe
import harmonic.model as md
import jax.numpy as jnp

domain = [np.array([1e-1, 1e1])]
sphere_1000D = mdl.HyperSphere(1000, domain)
sphere_2D = mdl.HyperSphere(2, domain)
real_nvp_2D = md.RealNVPModel(2)
spline_4D = md.RQSplineModel(4)

models_to_test_1 = [sphere_1000D, real_nvp_2D, spline_4D]
models_to_test_2 = [sphere_2D, real_nvp_2D, spline_4D]


@pytest.mark.parametrize("model", models_to_test_1)
def test_constructor(model):
    nchains = 100

    model.fitted = False
    with pytest.raises(ValueError):
        rho = cbe.Evidence(nchains=100, model=model)

    model.fitted = True
    with pytest.raises(ValueError):
        rho = cbe.Evidence(nchains=0, model=model)

    rho = cbe.Evidence(nchains, model)

    assert rho.nchains == nchains
    assert rho.evidence_inv == pytest.approx(0.0)
    assert rho.evidence_inv_var == pytest.approx(0.0)
    assert rho.evidence_inv_var_var == pytest.approx(0.0)
    assert rho.running_sum.size == nchains
    assert rho.nsamples_per_chain.size == nchains
    assert rho.shift_value == pytest.approx(0.0)
    assert rho.shift_set == False
    for i_chain in range(nchains):
        assert rho.running_sum[i_chain] == pytest.approx(0.0)
        assert rho.nsamples_per_chain[i_chain] == 0


@pytest.mark.parametrize("model", models_to_test_1)
def test_set_shift(model):
    nchains = 100

    model.fitted = True
    rho = cbe.Evidence(nchains, model)
    with pytest.raises(ValueError):
        rho.set_shift(np.nan)
    rho.set_shift(2.0)
    with pytest.raises(ValueError):
        rho.set_shift(1.0)
    assert rho.shift_value == pytest.approx(2.0)
    assert rho.shift_set == True


@pytest.mark.parametrize("model", models_to_test_1)
def test_process_run_with_shift(model):
    nchains = 10
    n_samples = 20

    model.fitted = True
    rho = cbe.Evidence(nchains, model)

    np.random.seed(1)
    samples = np.random.randn(nchains, n_samples)
    rho.running_sum = np.sum(samples, axis=1)
    rho.nsamples_per_chain = np.ones(nchains, dtype=int) * n_samples
    rho.process_run()

    evidence_inv = np.mean(samples)
    evidence_inv_var = np.std(np.sum(samples, axis=1) / n_samples) ** 2 / (nchains - 1)
    evidence_inv_var_var = (
        evidence_inv_var**2
        * (kurtosis(np.sum(samples, axis=1) / n_samples) + 2 + 2.0 / (nchains - 1))
        * (nchains - 1) ** 2
        / nchains**3
    )

    assert rho.evidence_inv == pytest.approx(evidence_inv, abs=1e-7)
    assert rho.evidence_inv_var == pytest.approx(evidence_inv_var)
    assert rho.evidence_inv_var_var == pytest.approx(evidence_inv_var_var)

    rho = cbe.Evidence(nchains, model, cbe.Shifting.MEAN_SHIFT)
    np.random.seed(1)
    post = np.random.uniform(high=1e3, size=(nchains, n_samples))
    samples = 1.0 / post
    mean_shift = np.mean(np.log(post))
    samples_scaled = samples * np.exp(mean_shift)
    rho.running_sum = np.sum(samples_scaled, axis=1)
    rho.nsamples_per_chain = np.ones(nchains, dtype=int) * n_samples
    rho.shift_value = mean_shift
    rho.process_run()

    evidence_inv = np.mean(samples)
    evidence_inv_var = np.std(np.sum(samples, axis=1) / n_samples) ** 2 / (nchains - 1)
    evidence_inv_var_var = (
        evidence_inv_var**2
        * (kurtosis(np.sum(samples, axis=1) / n_samples) + 2 + 2.0 / (nchains - 1))
        * (nchains - 1) ** 2
        / nchains**3
    )

    assert rho.evidence_inv == pytest.approx(evidence_inv, abs=1e-7)
    assert rho.evidence_inv_var == pytest.approx(evidence_inv_var)
    assert rho.evidence_inv_var_var == pytest.approx(evidence_inv_var_var)


def test_add_chains():
    nchains = 200
    nsamples = 500
    ndim = 2

    # Create samples of unnormalised Gaussian
    np.random.seed(30)
    X = np.random.randn(nchains, nsamples, ndim)
    Y = -np.sum(X * X, axis=2) / 2.0

    # Add samples to chains
    chain = ch.Chains(ndim)
    chain.add_chains_3d(X, Y)

    # Fit the Hyper_sphere
    domain = [np.array([1e-1, 1e1])]
    sphere = mdl.HyperSphere(ndim, domain)
    sphere.fit(chain.samples, chain.ln_posterior)

    # Calculate evidence
    cal_ev = cbe.Evidence(nchains, sphere, cbe.Shifting.MEAN_SHIFT)
    cal_ev.add_chains(chain)

    print("cal_ev.evidence_inv = {}".format(cal_ev.evidence_inv))

    assert cal_ev.evidence_inv == pytest.approx(0.159438606)
    assert cal_ev.evidence_inv_var == pytest.approx(1.164628268e-07)
    assert cal_ev.evidence_inv_var_var**0.5 == pytest.approx(1.142786462e-08)

    nsamples1 = 300
    chains1 = ch.Chains(ndim)
    for i_chain in range(nchains):
        chains1.add_chain(X[i_chain, :nsamples1, :], Y[i_chain, :nsamples1])
    chains2 = ch.Chains(ndim)
    for i_chain in range(nchains):
        chains2.add_chain(X[i_chain, nsamples1:, :], Y[i_chain, nsamples1:])

    ev = cbe.Evidence(nchains, sphere, cbe.Shifting.MEAN_SHIFT)
    # Might have small numerical differences if don't use same mean_shift.
    ev.add_chains(chains1)
    ev.add_chains(chains2)

    assert ev.evidence_inv == pytest.approx(0.159438606)
    assert ev.evidence_inv_var == pytest.approx(1.164628268e-07)
    assert ev.evidence_inv_var_var**0.5 == pytest.approx(1.142786462e-08)

    return


def test_shifting_settings():
    nchains = 200
    nsamples = 500
    ndim = 2

    # Create samples of unnormalised Gaussian
    np.random.seed(30)
    X = np.random.randn(nchains, nsamples, ndim)
    Y = -np.sum(X * X, axis=2) / 2.0

    # Add samples to chains
    chain = ch.Chains(ndim)
    chain.add_chains_3d(X, Y)

    # Fit the Hyper_sphere
    domain = [np.array([1e-1, 1e1])]
    sphere = mdl.HyperSphere(ndim, domain)
    sphere.fit(chain.samples, chain.ln_posterior)

    lnarg = np.zeros_like(chain.ln_posterior)

    # Check shift set correctly for: mean shift
    cal_ev = cbe.Evidence(nchains, sphere, cbe.Shifting.MEAN_SHIFT)
    cal_ev.add_chains(chain)

    for i_chains in range(nchains):
        i_samples_start = chain.start_indices[i_chains]
        i_samples_end = chain.start_indices[i_chains + 1]

        for i, i_samples in enumerate(range(i_samples_start, i_samples_end)):
            lnpred = cal_ev.model.predict(chain.samples[i_samples, :])
            lnprob = chain.ln_posterior[i_samples]
            lnarg[i_samples] = lnpred - lnprob
            if np.isinf(lnarg[i_samples]):
                lnarg[i_samples] = np.nan

    assert cal_ev.shift_value == pytest.approx(-np.nanmean(lnarg))

    # Check shift set correctly for: max shift
    cal_ev = cbe.Evidence(nchains, sphere, cbe.Shifting.MAX_SHIFT)
    cal_ev.add_chains(chain)
    assert cal_ev.shift_value == pytest.approx(-np.nanmax(lnarg))

    # Check shift set correctly for: min shift
    cal_ev = cbe.Evidence(nchains, sphere, cbe.Shifting.MIN_SHIFT)
    cal_ev.add_chains(chain)
    assert cal_ev.shift_value == pytest.approx(-np.nanmin(lnarg))

    # Check shift set correctly for: absmax shift
    cal_ev = cbe.Evidence(nchains, sphere, cbe.Shifting.ABS_MAX_SHIFT)
    cal_ev.add_chains(chain)
    assert cal_ev.shift_value == pytest.approx(-lnarg[np.nanargmax(np.abs(lnarg))])


@pytest.mark.parametrize("model", models_to_test_2)
def test_compute_evidence(model):
    nchains = 100

    model.fitted = True

    ev_inv = 1e10
    ev_inv_var = 2e10
    ev = cbe.Evidence(nchains, model)
    ev.evidence_inv = ev_inv
    ev.evidence_inv_var = ev_inv_var
    ev.ln_evidence_inv = np.log(ev_inv)
    ev.ln_evidence_inv_var = np.log(ev_inv_var)

    (evidence, evidence_std) = ev.compute_evidence()
    assert evidence == pytest.approx((1 + ev_inv_var / ev_inv**2) / ev_inv)
    assert evidence_std**2 == pytest.approx(ev_inv_var / ev_inv**4)

    (ln_evidence, ln_evidence_std) = ev.compute_ln_evidence()
    assert evidence == pytest.approx(np.exp(ln_evidence))
    assert evidence_std == pytest.approx(np.exp(ln_evidence_std))


@pytest.mark.parametrize("model", models_to_test_2)
def test_compute_ln_inv_evidence_errors(model):
    nchains = 100

    model.fitted = True

    # Check boundary case where ratio 1.0
    # (ln_ev_inv_var = 2 * ln_ev_inv)
    ln_ev_inv = 10
    ln_ev_inv_var = 2 * ln_ev_inv
    ev = cbe.Evidence(nchains, model)
    ev.evidence_inv = np.exp(ln_ev_inv)
    ev.evidence_inv_var = np.exp(ln_ev_inv_var)
    ev.ln_evidence_inv = ln_ev_inv
    ev.ln_evidence_inv_var = ln_ev_inv_var

    zeta_neg, zeta_pos = ev.compute_ln_inv_evidence_errors()
    assert zeta_neg == np.NINF
    assert zeta_pos == pytest.approx(np.log(2.0))

    # Check case where ln_ev_inv_var = ln_ev_inv
    ln_ev_inv = 10
    ln_ev_inv_var = ln_ev_inv
    ev = cbe.Evidence(nchains, model)
    ev.evidence_inv = np.exp(ln_ev_inv)
    ev.evidence_inv_var = np.exp(ln_ev_inv_var)
    ev.ln_evidence_inv = ln_ev_inv
    ev.ln_evidence_inv_var = ln_ev_inv_var

    zeta_neg, zeta_pos = ev.compute_ln_inv_evidence_errors()
    assert zeta_neg == pytest.approx(
        np.log(1.0 - np.exp(0.5 * ln_ev_inv_var - ln_ev_inv))
    )
    assert zeta_pos == pytest.approx(
        np.log(1.0 + np.exp(0.5 * ln_ev_inv_var - ln_ev_inv))
    )

    # Check case where ln_ev_inv_var = 0.5 * ln_ev_inv
    ln_ev_inv = 10
    ln_ev_inv_var = 0.5 * ln_ev_inv
    ev = cbe.Evidence(nchains, model)
    ev.evidence_inv = np.exp(ln_ev_inv)
    ev.evidence_inv_var = np.exp(ln_ev_inv_var)
    ev.ln_evidence_inv = ln_ev_inv
    ev.ln_evidence_inv_var = ln_ev_inv_var

    zeta_neg, zeta_pos = ev.compute_ln_inv_evidence_errors()
    assert zeta_neg == pytest.approx(
        np.log(1.0 - np.exp(0.5 * ln_ev_inv_var - ln_ev_inv))
    )
    assert zeta_pos == pytest.approx(
        np.log(1.0 + np.exp(0.5 * ln_ev_inv_var - ln_ev_inv))
    )


def test_compute_bayes_factors():
    ndim = 2
    nchains = 100

    domain = [np.array([1e-1, 1e1])]
    sphere = mdl.HyperSphere(ndim, domain)
    sphere.fitted = True

    ev1_inv = 1e10
    ev1_inv_var = 2e10
    ev1 = cbe.Evidence(nchains, sphere)
    ev1.evidence_inv = ev1_inv
    ev1.evidence_inv_var = ev1_inv_var
    ev1.chains_added = True

    ndim = 4
    nchains = 600

    ev2_inv = 3e10
    ev2_inv_var = 4e10
    ev2 = cbe.Evidence(nchains, sphere)
    ev2.evidence_inv = ev2_inv
    ev2.evidence_inv_var = ev2_inv_var
    ev2.chains_added = True

    bf12_check = ev2_inv / ev1_inv * (1.0 + ev2_inv_var / ev2_inv**2)
    bf12_var_check = (
        ev1_inv**2 * ev2_inv_var + ev2_inv**2 * ev1_inv_var
    ) / ev1_inv**4

    (bf12, bf12_std) = cbe.compute_bayes_factor(ev1, ev2)

    assert bf12 == pytest.approx(bf12_check)
    assert bf12_std == pytest.approx(np.sqrt(bf12_var_check))

    (ln_bf12, ln_bf12_std) = cbe.compute_ln_bayes_factor(ev1, ev2)

    assert bf12 == pytest.approx(np.exp(ln_bf12))
    assert bf12_std == pytest.approx(np.exp(ln_bf12_std))

    # Test bayes factor reduces to single evidence calculation.
    ev2_inv = 1.0
    ev2_inv_var = 0.0
    ev2 = cbe.Evidence(nchains, sphere)
    ev2.evidence_inv = ev2_inv
    ev2.evidence_inv_var = ev2_inv_var
    ev2.chains_added = True
    (bf12, bf12_std) = cbe.compute_bayes_factor(ev1, ev2)

    (evidence, evidence_std) = ev1.compute_evidence()
    assert bf12 == pytest.approx(evidence)
    assert bf12_std == pytest.approx(evidence_std)


@pytest.mark.parametrize("model", models_to_test_2)
def test_serialization(model):
    nchains = 200
    nsamples = 500
    ndim = model.ndim

    # Create samples of unnormalised Gaussian
    np.random.seed(30)
    X = np.random.randn(nchains, nsamples, ndim)
    Y = -np.sum(X * X, axis=2) / 2.0

    # Add samples to chains
    chain = ch.Chains(ndim)
    chain.add_chains_3d(X, Y)

    # Fit the model
    if not hasattr(model, "flow"):
        model.fit(chain.samples, chain.ln_posterior)
    else:
        model.fit(chain.samples, epochs=5)

    # Set up the evidence object
    ev1 = cbe.Evidence(nchains, model)
    ev1.add_chains(chain)

    # Serialize evidence
    ev1.serialize(".test.dat")

    # Deserialize evidence
    ev2 = cbe.Evidence.deserialize(".test.dat")

    # Test evidence objects the same
    assert ev1.nchains == ev2.nchains
    assert ev1.evidence_inv == ev2.evidence_inv
    assert ev1.evidence_inv_var == ev2.evidence_inv_var
    assert ev1.evidence_inv_var_var == ev2.evidence_inv_var_var
    assert ev1.running_sum.size == ev2.running_sum.size
    assert ev1.nsamples_per_chain.size == ev2.nsamples_per_chain.size
    assert ev1.shift_value == ev2.shift_value
    assert ev1.shift_set == ev2.shift_set
    for i_chain in range(nchains):
        assert ev1.running_sum[i_chain] == ev2.running_sum[i_chain]
        assert ev1.nsamples_per_chain[i_chain] == ev2.nsamples_per_chain[i_chain]
    if not hasattr(model, "flow"):
        test = np.ones(ndim)
    else:
        test = np.array([np.ones(ndim)])
    assert ev1.model.predict(test) == ev2.model.predict(test)


def test_n_eff():
    """Test calculation of effective sample size for chains with equal and unequal number of samples."""
    nchains1 = 5
    nsamples1 = 500
    nchains2 = 5
    nsamples2 = 20
    ndim = 2

    # Create samples of unnormalised Gaussian
    np.random.seed(30)
    X1 = np.random.randn(nchains1, nsamples1, ndim)
    Y1 = -np.sum(X1 * X1, axis=2) / 2.0
    X2 = np.random.randn(nchains2, nsamples2, ndim)
    Y2 = -np.sum(X2 * X2, axis=2) / 2.0

    # Add samples to chains
    chain_equal = ch.Chains(ndim)  # chains with equal number of samples per chain
    chain_equal.add_chains_3d(X1, Y1)
    chain_equal.add_chains_3d(X1, Y1)

    chain_unequal = ch.Chains(ndim)  # chains with unequal number of samples per chain
    chain_unequal.add_chains_3d(X1, Y1)
    chain_unequal.add_chains_3d(X2, Y2)

    # Fit the Hyper_sphere
    domain = [np.array([1e-1, 1e1])]
    sphere_equal = mdl.HyperSphere(ndim, domain)
    sphere_equal.fit(chain_equal.samples, chain_equal.ln_posterior)
    sphere_unequal = mdl.HyperSphere(ndim, domain)
    sphere_unequal.fit(chain_unequal.samples, chain_unequal.ln_posterior)

    # Set up the evidence object
    ev_equal = cbe.Evidence(nchains1 * 2, sphere_equal)
    ev_equal.add_chains(chain_equal)

    ev_unequal = cbe.Evidence(nchains1 + nchains2, sphere_unequal)
    ev_unequal.add_chains(chain_unequal)

    assert ev_equal.n_eff == sum(ev_equal.nsamples_per_chain) ** 2 / sum(
        ev_equal.nsamples_per_chain**2
    ), "Effective sample size calculation is incorrect for equal number of samples in the chains."
    assert ev_unequal.n_eff == sum(ev_unequal.nsamples_per_chain) ** 2 / sum(
        ev_unequal.nsamples_per_chain**2
    ), "Effective sample size calculation is incorrect for unequal number of samples in the chains."


@pytest.mark.parametrize("model", models_to_test_2)
def test_nsamples_per_chain(model):
    nchains = 3
    nsamples = 5
    ndim = model.ndim

    # Create mock samples
    np.random.seed(30)
    X = np.zeros((nchains, nsamples, ndim))
    # Introduce NaN in first chain
    X[0, 0, 0] = np.nan
    Y = np.ones((nchains, nsamples))

    # Add samples to chains
    chain = ch.Chains(ndim)
    chain.add_chains_3d(X, Y)

    model.fitted = True
    ev = cbe.Evidence(nchains, model)
    ev.add_chains(chain)

    nsamples_per_chain_ref = np.full((nchains), nsamples)
    nsamples_eff_per_chain_ref = np.full((nchains), nsamples)
    nsamples_eff_per_chain_ref[0] -= 1

    for i_chain in range(nchains):
        assert ev.nsamples_per_chain[i_chain] == nsamples_per_chain_ref[i_chain], (
            "Number of samples per chain is "
            + str(ev.nsamples_per_chain[i_chain])
            + " instead of "
            + str(nsamples_per_chain_ref[i_chain])
        )
        assert (
            ev.nsamples_eff_per_chain[i_chain] == nsamples_eff_per_chain_ref[i_chain]
        ), (
            "Effective number of samples per chain is "
            + str(ev.nsamples_eff_per_chain[i_chain])
            + " instead of "
            + str(nsamples_eff_per_chain_ref[i_chain])
        )
