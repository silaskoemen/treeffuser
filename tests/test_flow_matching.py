import numpy as np
import pytest

from treeffuser import Treeffuser
from treeffuser._flow_matching import LinearFlowPath
from treeffuser._flow_matching import ReverseVelocityInterpolant
from treeffuser._flow_matching import ReverseVelocityODE
from treeffuser._flow_matching import TrigFlowPath
from treeffuser._flow_matching import VPFlowPath
from treeffuser._flow_matching import get_flow_path
from treeffuser._flow_matching import get_stochasticity_schedule
from treeffuser._flow_matching import linear_stochasticity_schedule
from treeffuser._score_models import _FLOW_MATCHING_T_EPS
from treeffuser._score_models import LightGBMVelocityModel
from treeffuser._score_models import LogBetaNormalFlowMatchingTSampler
from treeffuser._score_models import LogSNRNormalFlowMatchingTSampler
from treeffuser._score_models import _make_flow_matching_training_data
from treeffuser._score_models import get_flow_matching_t_sampler
from treeffuser.sde import sdeint


def test_linear_flow_path_boundaries_and_velocity():
    path = LinearFlowPath()
    y0 = np.array([[1.0, -2.0], [0.5, 3.0]])
    z = np.array([[0.2, 0.4], [-1.0, 1.5]])
    t0 = np.zeros((2, 1))
    t1 = np.ones((2, 1))
    tm = np.array([[0.25], [0.75]])

    assert path.name == "linear"
    assert get_flow_path("linear").name == "linear"
    assert np.allclose(path.interpolate(y0=y0, z=z, t=t0), y0)
    assert np.allclose(path.interpolate(y0=y0, z=z, t=t1), z)
    assert np.allclose(path.interpolate(y0=y0, z=z, t=tm), (1.0 - tm) * y0 + tm * z)
    assert np.allclose(path.target_velocity(y0=y0, z=z, t=tm), z - y0)
    with pytest.raises(ValueError, match="Unknown flow path"):
        get_flow_path("bogus")


def test_make_flow_matching_training_data_shapes_endpoint_and_cat_idx():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=(30, 2))
    cat_idx = [1, 3]
    path = LinearFlowPath()

    baseline = _make_flow_matching_training_data(
        X=X,
        y=y,
        flow_path=path,
        n_repeats=2,
        eval_percent=0.2,
        cat_idx=list(cat_idx),
        seed=0,
    )
    repeat = _make_flow_matching_training_data(
        X=X,
        y=y,
        flow_path=path,
        n_repeats=2,
        eval_percent=0.2,
        cat_idx=list(cat_idx),
        seed=0,
    )
    predictors_train, predictors_val, target_train, target_val, transformed_cat_idx = baseline

    assert predictors_val is not None
    assert target_val is not None
    assert predictors_train.shape[1] == y.shape[1] + X.shape[1] + 1
    assert predictors_val.shape[1] == y.shape[1] + X.shape[1] + 1
    assert target_train.shape[1] == y.shape[1]
    assert target_val.shape[1] == y.shape[1]
    assert transformed_cat_idx == [c + y.shape[1] for c in cat_idx]
    assert predictors_train[:, -1].min() >= 1e-5
    assert predictors_train[:, -1].max() == 1.0
    for observed, expected in zip(baseline, repeat, strict=True):
        assert np.array_equal(observed, expected)


def test_reverse_velocity_ode_constant_velocity_reaches_data_in_one_heun_step():
    y0 = np.array([[1.0], [-2.0]])
    z = np.array([[4.0], [3.0]])
    constant_velocity = z - y0
    ode = ReverseVelocityODE(velocity_fn=lambda y, t: np.broadcast_to(constant_velocity, y.shape))

    samples = sdeint(ode, z, 0.0, 1.0, n_steps=1, method="heun", seed=0)

    assert np.allclose(samples, y0)


def test_lightgbm_velocity_model_runs_and_predicts_finite_velocity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = X[:, :1] - X[:, 1:] + rng.normal(scale=0.1, size=(120, 1))
    model = LightGBMVelocityModel(
        n_repeats=2,
        n_estimators=20,
        early_stopping_rounds=None,
        learning_rate=0.1,
        eval_percent=None,
        seed=0,
        verbose=-1,
    )

    model.fit(X, y)
    velocity = model.velocity(y=y[:10], X=X[:10], t=np.ones((10, 1)))

    assert model.flow_path.name == "linear"
    assert velocity.shape == y[:10].shape
    assert np.all(np.isfinite(velocity))


def test_treeffuser_flow_matching_end_to_end_and_chunk_seeds():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(160, 2))
    y = X[:, :1] + rng.normal(scale=0.2, size=(160, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        flow_path="linear",
        n_repeats=2,
        n_estimators=30,
        early_stopping_rounds=None,
        learning_rate=0.1,
        eval_percent=None,
        seed=0,
        verbose=-1,
    )

    model.fit(X, y)
    samples = model.sample(X[:5], n_samples=5, n_parallel=2, n_steps=8, sampler_method="heun", seed=10)
    samples_redo = model.sample(X[:5], n_samples=5, n_parallel=2, n_steps=8, sampler_method="heun", seed=10)

    assert samples.shape == (5, 5, 1)
    assert np.all(np.isfinite(samples))
    assert np.allclose(samples, samples_redo)
    assert not np.allclose(samples[:2], samples[2:4])
    assert model.n_estimators_true == model.velocity_model.n_estimators_true


def test_treeffuser_flow_matching_residualization_end_to_end():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 2))
    y = X[:, :1] - X[:, 1:] + rng.normal(scale=0.2, size=(120, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        residualize="mean",
        residualize_k_folds=3,
        extra_residualizer_params={
            "n_estimators": 5,
            "max_depth": 2,
            "num_leaves": 3,
            "min_child_samples": 5,
        },
        n_repeats=1,
        n_estimators=10,
        early_stopping_rounds=None,
        eval_percent=None,
        seed=0,
        verbose=-1,
    )

    model.fit(X, y)
    samples = model.sample(X[:4], n_samples=3, n_parallel=2, n_steps=4, sampler_method="heun", seed=0)

    assert model._residualizer is not None
    assert samples.shape == (3, 4, 1)
    assert np.all(np.isfinite(samples))


def test_treeffuser_flow_matching_rejects_pf_ode_flag():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 1))
    y = X + rng.normal(scale=0.1, size=(80, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        n_repeats=1,
        n_estimators=5,
        early_stopping_rounds=None,
        eval_percent=None,
        seed=0,
        verbose=-1,
    ).fit(X, y)

    with pytest.raises(ValueError, match="pf_ode=True"):
        model.sample(X[:2], n_samples=2, n_steps=2, sampler_method="heun", pf_ode=True, seed=0)


def test_treeffuser_flow_matching_warns_on_score_only_params():
    # t_sampling is shared with FM (log-beta-normal sampler reuses the name),
    # but score_parameterization and sde_hyperparam_max are still score-only.
    with pytest.warns(UserWarning, match="ignores score/SDE-only parameters") as warnings_record:
        Treeffuser(
            training_objective="flow_matching",
            score_parameterization="edm",
            sde_hyperparam_max=20.0,
        )

    message = str(warnings_record[0].message)
    assert "score_parameterization" in message
    assert "sde_hyperparam_max" in message
    assert "t_sampling" not in message


def test_treeffuser_flow_matching_gaussian_smoke_distribution():
    rng = np.random.default_rng(3)
    X = np.zeros((700, 1))
    y = rng.normal(loc=0.0, scale=1.0, size=(700, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        n_repeats=3,
        n_estimators=80,
        early_stopping_rounds=None,
        learning_rate=0.05,
        eval_percent=None,
        min_child_samples=5,
        seed=0,
        verbose=-1,
    )

    model.fit(X, y)
    samples = model.sample(X[:80], n_samples=4, n_parallel=2, n_steps=16, sampler_method="heun", seed=20)
    flat = samples.reshape(-1)

    assert abs(float(flat.mean())) < 0.3
    assert 0.6 < float(flat.std()) < 1.4


def test_linear_flow_path_implied_score_matches_closed_form():
    # At t=1 the marginal is N(0, I), so score at y_t = z under v = z - 0 = z is -z.
    path = LinearFlowPath()
    z = np.array([[1.5], [-0.5], [2.0]])
    t1 = np.ones((3, 1))
    v_at_prior = z  # E[z | y_t = z] - 0 = z
    score = path.implied_score(y_t=z, velocity=v_at_prior, t=t1)
    assert np.allclose(score, -z)

    # Generic midpoint check: score = -(y_t + (1-t)v)/t.
    y_t = np.array([[0.3, -0.2], [1.1, 0.0]])
    v = np.array([[1.0, -0.5], [0.2, 0.7]])
    t = np.array([[0.25], [0.75]])
    expected = -(y_t + (1.0 - t) * v) / t
    assert np.allclose(path.implied_score(y_t=y_t, velocity=v, t=t), expected)


def test_reverse_velocity_interpolant_zero_stochasticity_matches_ode():
    # ε(t) ≡ 0 must reduce exactly to ReverseVelocityODE on the same velocity field.
    path = LinearFlowPath()
    rng = np.random.default_rng(7)
    z = rng.normal(size=(8, 1))

    def velocity_fn(y, t):
        # Arbitrary smooth toy velocity; sign-test alone doesn't depend on its form.
        return -0.3 * y + 0.1 * t

    ode = ReverseVelocityODE(velocity_fn=velocity_fn)
    interp = ReverseVelocityInterpolant(
        velocity_fn=velocity_fn,
        flow_path=path,
        stochasticity_schedule=linear_stochasticity_schedule(0.0),
    )

    ode_samples = sdeint(ode, z, 0.0, 1.0 - 1e-5, n_steps=20, method="heun", seed=42)
    interp_samples = sdeint(interp, z, 0.0, 1.0 - 1e-5, n_steps=20, method="heun", seed=42)
    assert np.allclose(ode_samples, interp_samples)


def test_reverse_velocity_interpolant_changes_samples_under_positive_stochasticity():
    path = LinearFlowPath()
    z = np.array([[1.0], [-1.0], [0.5], [0.2]])

    def velocity_fn(y, t):
        return -0.3 * y

    det = sdeint(
        ReverseVelocityODE(velocity_fn=velocity_fn),
        z,
        0.0,
        1.0 - 1e-5,
        n_steps=20,
        method="heun",
        seed=0,
    )
    stoch = sdeint(
        ReverseVelocityInterpolant(
            velocity_fn=velocity_fn,
            flow_path=path,
            stochasticity_schedule=linear_stochasticity_schedule(0.5),
        ),
        z,
        0.0,
        1.0 - 1e-5,
        n_steps=20,
        method="heun",
        seed=0,
    )
    assert not np.allclose(det, stoch)
    assert np.all(np.isfinite(stoch))


def test_treeffuser_stochastic_fm_recovers_unit_gaussian_marginal():
    # ε > 0 should not destroy the smoke-test recovery of N(0, 1).
    rng = np.random.default_rng(11)
    X = np.zeros((700, 1))
    y = rng.normal(loc=0.0, scale=1.0, size=(700, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        n_repeats=3,
        n_estimators=80,
        early_stopping_rounds=None,
        learning_rate=0.05,
        eval_percent=None,
        min_child_samples=5,
        seed=0,
        verbose=-1,
    ).fit(X, y)

    samples = model.sample(
        X[:80],
        n_samples=4,
        n_parallel=2,
        n_steps=24,
        sampler_method="heun",
        seed=20,
        velocity_stochasticity=0.5,
    )
    flat = samples.reshape(-1)
    assert abs(float(flat.mean())) < 0.3
    assert 0.6 < float(flat.std()) < 1.5
    assert np.all(np.isfinite(flat))


def test_velocity_stochasticity_rejected_for_score_training():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(60, 1))
    y = X + rng.normal(scale=0.1, size=(60, 1))
    model = Treeffuser(
        n_repeats=1,
        n_estimators=10,
        early_stopping_rounds=None,
        eval_percent=None,
        seed=0,
        verbose=-1,
    ).fit(X, y)
    with pytest.raises(ValueError, match="velocity_stochasticity is only valid"):
        model.sample(X[:2], n_samples=2, n_steps=2, seed=0, velocity_stochasticity=0.5)


def test_velocity_stochasticity_rejected_when_negative():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(60, 1))
    y = X + rng.normal(scale=0.1, size=(60, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        n_repeats=1,
        n_estimators=10,
        early_stopping_rounds=None,
        eval_percent=None,
        seed=0,
        verbose=-1,
    ).fit(X, y)
    with pytest.raises(ValueError, match="velocity_stochasticity must be non-negative"):
        model.sample(X[:2], n_samples=2, n_steps=2, sampler_method="heun", seed=0, velocity_stochasticity=-0.1)


def test_stochasticity_schedule_shapes_have_correct_endpoints_and_peaks():
    # Check the four schedule shapes at a grid of t values.
    t_grid = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    c = 0.7
    expected = {
        "linear": c * t_grid,
        "quadratic": c * t_grid**2,
        "sqrt": c * np.sqrt(t_grid),
        "tent": c * t_grid * (1.0 - t_grid),
    }
    for shape, want in expected.items():
        schedule = get_stochasticity_schedule(shape, c)
        got = schedule(t_grid)
        assert np.allclose(got, want), f"shape={shape} mismatch"
    # Tent must peak at t=0.5 and vanish at both endpoints.
    tent = get_stochasticity_schedule("tent", 1.0)
    assert np.isclose(tent(np.array([[0.0]])), 0.0)
    assert np.isclose(tent(np.array([[1.0]])), 0.0)
    assert np.isclose(tent(np.array([[0.5]])), 0.25)
    # Unknown shape rejected.
    with pytest.raises(ValueError, match="Unknown stochasticity schedule shape"):
        get_stochasticity_schedule("bogus", 1.0)


def test_stochasticity_schedule_threaded_through_treeffuser_sample():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 1))
    y = X + rng.normal(scale=0.1, size=(80, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        n_repeats=2,
        n_estimators=15,
        early_stopping_rounds=None,
        eval_percent=None,
        seed=0,
        verbose=-1,
    ).fit(X, y)

    # Different shapes at fixed strength must produce visibly different samples.
    linear = model.sample(
        X[:5],
        n_samples=4,
        n_parallel=2,
        n_steps=8,
        sampler_method="heun",
        seed=10,
        velocity_stochasticity=0.5,
        velocity_stochasticity_schedule="linear",
    )
    quadratic = model.sample(
        X[:5],
        n_samples=4,
        n_parallel=2,
        n_steps=8,
        sampler_method="heun",
        seed=10,
        velocity_stochasticity=0.5,
        velocity_stochasticity_schedule="quadratic",
    )
    assert not np.allclose(linear, quadratic)
    assert np.all(np.isfinite(linear))
    assert np.all(np.isfinite(quadratic))

    # Stochasticity=0 makes the schedule choice a no-op.
    det_linear = model.sample(
        X[:5],
        n_samples=4,
        n_parallel=2,
        n_steps=8,
        sampler_method="heun",
        seed=10,
        velocity_stochasticity=0.0,
        velocity_stochasticity_schedule="linear",
    )
    det_tent = model.sample(
        X[:5],
        n_samples=4,
        n_parallel=2,
        n_steps=8,
        sampler_method="heun",
        seed=10,
        velocity_stochasticity=0.0,
        velocity_stochasticity_schedule="tent",
    )
    assert np.allclose(det_linear, det_tent)


def test_trig_flow_path_boundaries_and_finite_difference_velocity():
    path = TrigFlowPath()
    y0 = np.array([[1.5, -0.5], [2.0, 3.0]])
    z = np.array([[0.3, 0.1], [-1.0, 0.5]])
    t0 = np.zeros((2, 1))
    t1 = np.ones((2, 1))
    # Boundaries: y(0) = y0, y(1) = z exactly.
    assert np.allclose(path.interpolate(y0, z, t0), y0)
    assert np.allclose(path.interpolate(y0, z, t1), z)
    # Variance preserving: a^2 + b^2 = 1.
    t_grid = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    a, b, _, _ = path._coeffs(t_grid)
    assert np.allclose(a**2 + b**2, 1.0)
    # Velocity matches central finite-difference of interpolate.
    t = np.array([[0.3], [0.7]])
    h = 1e-6
    y0_pair, z_pair = y0[:2], z[:2]
    fd = (path.interpolate(y0_pair, z_pair, t + h) - path.interpolate(y0_pair, z_pair, t - h)) / (2 * h)
    assert np.allclose(fd, path.target_velocity(y0_pair, z_pair, t), atol=1e-4)


def test_trig_implied_score_closed_form_matches_implementation():
    path = TrigFlowPath()
    y_t = np.array([[0.3, 1.2], [-1.0, 0.4]])
    v = np.array([[0.7, -0.2], [1.1, 0.0]])
    t = np.array([[0.25], [0.75]])
    phi = np.pi * t / 2.0
    expected = -y_t - (2.0 / np.pi) * (np.cos(phi) / np.sin(phi)) * v
    assert np.allclose(path.implied_score(y_t, v, t), expected)
    # At t=1 the path reaches the prior; score(z, 1) = -z under v = z - 0 (residualized).
    z = np.array([[0.5], [-1.0]])
    score_at_prior = path.implied_score(z, z, np.ones((2, 1)))
    # With v = z and t=1: phi=pi/2, cos=0, cot=0, score = -z.
    assert np.allclose(score_at_prior, -z)


def test_vp_flow_path_boundaries_variance_preserving_and_finite_difference():
    path = VPFlowPath()
    y0 = np.array([[1.5, -0.5], [2.0, 3.0]])
    z = np.array([[0.3, 0.1], [-1.0, 0.5]])
    t0 = np.zeros((2, 1))
    # y(0) = y0 exactly (a(0) = 1, b(0) = 0).
    assert np.allclose(path.interpolate(y0, z, t0), y0)
    # Variance preserving across all t.
    t_grid = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    a, b, _, _ = path._coeffs(t_grid)
    assert np.allclose(a**2 + b**2, 1.0, atol=1e-8)
    # a(1) is small but nonzero (~0.08 with default betas); b(1) ~ 1.
    a1, b1, _, _ = path._coeffs(np.ones((1, 1)))
    assert 0.0 < a1[0, 0] < 0.15
    assert 0.99 < b1[0, 0] < 1.0
    # Velocity matches finite-difference at interior points.
    t = np.array([[0.3], [0.7]])
    h = 1e-6
    y0_pair, z_pair = y0[:2], z[:2]
    fd = (path.interpolate(y0_pair, z_pair, t + h) - path.interpolate(y0_pair, z_pair, t - h)) / (2 * h)
    assert np.allclose(fd, path.target_velocity(y0_pair, z_pair, t), atol=1e-4)


def test_vp_rejects_invalid_beta_settings():
    with pytest.raises(ValueError, match="0 < beta_min < beta_max"):
        VPFlowPath(beta_min=0.0)
    with pytest.raises(ValueError, match="0 < beta_min < beta_max"):
        VPFlowPath(beta_min=20.0, beta_max=10.0)


def test_get_flow_path_recognises_new_names():
    assert isinstance(get_flow_path("trig"), TrigFlowPath)
    assert isinstance(get_flow_path("vp"), VPFlowPath)
    assert isinstance(get_flow_path("linear"), LinearFlowPath)
    # Passing an instance is idempotent.
    p = TrigFlowPath()
    assert get_flow_path(p) is p


def test_treeffuser_trig_path_end_to_end_gaussian_smoke():
    rng = np.random.default_rng(101)
    X = np.zeros((700, 1))
    y = rng.normal(loc=0.0, scale=1.0, size=(700, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        flow_path="trig",
        n_repeats=3,
        n_estimators=80,
        early_stopping_rounds=None,
        learning_rate=0.05,
        eval_percent=None,
        min_child_samples=5,
        seed=0,
        verbose=-1,
    ).fit(X, y)
    samples = model.sample(
        X[:80],
        n_samples=4,
        n_parallel=2,
        n_steps=16,
        sampler_method="heun",
        seed=20,
    )
    flat = samples.reshape(-1)
    assert abs(float(flat.mean())) < 0.3
    assert 0.6 < float(flat.std()) < 1.4


def test_treeffuser_vp_path_end_to_end_gaussian_smoke():
    rng = np.random.default_rng(103)
    X = np.zeros((700, 1))
    y = rng.normal(loc=0.0, scale=1.0, size=(700, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        flow_path="vp",
        n_repeats=3,
        n_estimators=80,
        early_stopping_rounds=None,
        learning_rate=0.05,
        eval_percent=None,
        min_child_samples=5,
        seed=0,
        verbose=-1,
    ).fit(X, y)
    samples = model.sample(
        X[:80],
        n_samples=4,
        n_parallel=2,
        n_steps=24,
        sampler_method="heun",
        seed=20,
    )
    flat = samples.reshape(-1)
    # VP has higher-variance velocity targets near t=0; recovery is slightly noisier.
    assert abs(float(flat.mean())) < 0.4
    assert 0.5 < float(flat.std()) < 1.6


def test_flow_path_noise_scale_endpoints():
    # noise_scale(0) = 0, noise_scale(1) = 1 (exactly for linear and trig; ~1 for VP).
    for path_cls in [LinearFlowPath, TrigFlowPath]:
        p = path_cls()
        assert np.isclose(p.noise_scale(np.zeros((1, 1)))[0, 0], 0.0)
        assert np.isclose(p.noise_scale(np.ones((1, 1)))[0, 0], 1.0)
    p = VPFlowPath()
    assert np.isclose(p.noise_scale(np.zeros((1, 1)))[0, 0], 0.0)
    assert 0.99 < p.noise_scale(np.ones((1, 1)))[0, 0] < 1.0


def test_log_beta_normal_fm_t_sampler_concentrates_mass_per_p_mean():
    rng = np.random.default_rng(0)
    sampler = get_flow_matching_t_sampler("log_sigma_normal", log_sigma_p_mean=-1.2, log_sigma_p_std=1.2)
    assert isinstance(sampler, LogBetaNormalFlowMatchingTSampler)

    # For linear FM, beta(t)=t so log beta = log t. Mean log beta should track p_mean.
    path = LinearFlowPath()
    t = sampler.sample(10_000, path, rng)
    assert t.shape == (10_000, 1)
    assert t.min() >= 1e-5
    assert t.max() <= 1.0
    # Median should sit near exp(-1.2) ~ 0.30 (log beta = log t = p_mean median).
    assert 0.2 < float(np.median(t)) < 0.4

    # For VP, log beta is path-dependent but the mass should still be concentrated.
    vp = VPFlowPath()
    t_vp = sampler.sample(10_000, vp, rng)
    assert t_vp.shape == (10_000, 1)
    assert np.all(np.isfinite(t_vp))


def test_treeffuser_fm_with_log_sigma_t_sampling_runs_end_to_end():
    rng = np.random.default_rng(11)
    X = np.zeros((400, 1))
    y = rng.normal(loc=0.0, scale=1.0, size=(400, 1))
    model = Treeffuser(
        training_objective="flow_matching",
        flow_path="vp",
        t_sampling="log_sigma_normal",
        log_sigma_p_mean=-1.2,
        log_sigma_p_std=1.2,
        n_repeats=2,
        n_estimators=50,
        early_stopping_rounds=None,
        eval_percent=None,
        min_child_samples=5,
        seed=0,
        verbose=-1,
    )
    model.fit(X, y)
    samples = model.sample(X[:30], n_samples=4, n_parallel=2, n_steps=8, sampler_method="heun", seed=20)
    assert samples.shape == (4, 30, 1)
    assert np.all(np.isfinite(samples))


def test_log_snr_sampler_no_clipping_with_default_params():
    # log SNR ranges over the full real line, so a Normal centred near 0 with std 2
    # should not produce a degenerate spike at the endpoints. This is the key
    # property that distinguishes it from LogBetaNormal, which clips at t=1.
    rng = np.random.default_rng(0)
    sampler = LogSNRNormalFlowMatchingTSampler(p_mean=0.0, p_std=2.0)
    path = LinearFlowPath()
    t = sampler.sample(20_000, path, rng)
    # No more than a few percent of draws should hit the t=1 endpoint exactly
    # (would indicate clipping); contrast with LogBeta default which clips ~16%.
    endpoint_fraction = float(np.mean(t >= 1.0 - 1e-9))
    assert endpoint_fraction < 0.02


def test_uniform_endpoint_fraction_threads_through_treeffuser():
    # Endpoint fraction = 0.5 should make ~half of sampled t values equal to 1.0.
    model = Treeffuser(
        training_objective="flow_matching",
        t_sampling="uniform",
        uniform_endpoint_fraction=0.5,
        n_repeats=1,
        n_estimators=5,
        early_stopping_rounds=None,
        eval_percent=None,
        seed=0,
        verbose=-1,
    )
    assert model.uniform_endpoint_fraction == 0.5
    # Sanity: get_new_velocity_model wires the parameter through.
    vm = model.get_new_velocity_model()
    rng_v = np.random.default_rng(0)
    t_samples = vm.t_sampler.sample(2000, LinearFlowPath(), rng_v)
    anchor_rate = float(np.mean(t_samples >= 1.0 - 1e-9))
    # Allow noise; the expected rate is 0.5.
    assert 0.4 < anchor_rate < 0.6
    assert t_samples.min() >= _FLOW_MATCHING_T_EPS
