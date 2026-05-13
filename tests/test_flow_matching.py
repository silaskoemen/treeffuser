import numpy as np
import pytest

from treeffuser import Treeffuser
from treeffuser._flow_matching import LinearFlowPath
from treeffuser._flow_matching import ReverseVelocityInterpolant
from treeffuser._flow_matching import ReverseVelocityODE
from treeffuser._flow_matching import get_flow_path
from treeffuser._flow_matching import linear_stochasticity_schedule
from treeffuser._score_models import LightGBMVelocityModel
from treeffuser._score_models import _make_flow_matching_training_data
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
    with pytest.warns(UserWarning, match="ignores score/SDE-only parameters") as warnings_record:
        Treeffuser(
            training_objective="flow_matching",
            score_parameterization="edm",
            t_sampling="log_sigma_normal",
            sde_hyperparam_max=20.0,
        )

    message = str(warnings_record[0].message)
    assert "score_parameterization" in message
    assert "t_sampling" in message
    assert "sde_hyperparam_max" in message


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
