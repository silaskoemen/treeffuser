"""
Contains all of the test for the different score model classes.
"""

import numpy as np
import pytest
from einops import repeat

from treeffuser._score_models import LightGBMScoreModel
from treeffuser._score_models import NoiseParameterization
from treeffuser._score_models import RawTimeFeatureBuilder
from treeffuser._score_models import RawTimeLogStdFeatureBuilder
from treeffuser._score_models import X0Parameterization
from treeffuser._score_models import _make_training_data
from treeffuser._score_models import get_noise_feature_builder
from treeffuser._score_models import get_score_parameterization
from treeffuser.sde.diffusion_sdes import VESDE
from treeffuser.sde.diffusion_sdes import VPSDE

from .utils import generate_bimodal_linear_regression_data
from .utils import r2_score


def test_noise_parameterization_matches_current_behavior():
    parameterization = NoiseParameterization()
    z = np.array([[1.0, -2.0], [0.5, 4.0]])
    std = np.array([0.25, 2.0])
    prediction = np.array([-0.5, 1.0])
    t = np.array([[0.1], [0.2]])

    target = parameterization.make_target(
        y0=np.zeros_like(z),
        perturbed_y=np.zeros_like(z),
        z=z,
        mean=np.zeros_like(z),
        std=np.ones_like(z),
        t=t,
    )
    score = parameterization.reconstruct_score(
        prediction=prediction.reshape(-1, 1),
        perturbed_y=np.zeros_like(prediction).reshape(-1, 1),
        std=std.reshape(-1, 1),
    )

    assert np.allclose(target, -z)
    assert np.allclose(score[:, 0], prediction / std)


def test_x0_parameterization_reconstructs_score_from_denoised_prediction():
    parameterization = X0Parameterization()
    y0 = np.array([[1.0, -2.0], [0.5, 4.0]])
    t = np.array([[0.1], [0.2]])
    z = np.array([[0.3, -0.4], [1.2, -0.7]])
    sde = VPSDE(hyperparam_min=0.1, hyperparam_max=1.0)
    mean, std = sde.get_mean_std_pt_given_y0(y0, t)
    perturbed_y = mean + std * z

    target = parameterization.make_target(
        y0=y0,
        perturbed_y=perturbed_y,
        z=z,
        mean=mean,
        std=std,
        t=t,
    )
    score = parameterization.reconstruct_score(
        prediction=y0,
        perturbed_y=perturbed_y,
        std=std,
        predicted_mean=mean,
    )
    expected_score = (mean - perturbed_y) / (std**2)

    assert parameterization.name == "x0"
    assert get_score_parameterization("x0").name == "x0"
    assert np.array_equal(target, y0)
    assert np.allclose(score, expected_score)


def test_raw_time_feature_builder_matches_concatenation():
    perturbed_y = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    t = np.array([[0.1], [0.5], [0.9]])
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=1.0)

    builder = RawTimeFeatureBuilder()
    features = builder.make_features(perturbed_y=perturbed_y, X=X, t=t, sde=sde)

    expected = np.concatenate([perturbed_y, X, t], axis=1)
    assert features.shape == expected.shape
    assert np.array_equal(features, expected)


def test_raw_time_log_std_feature_builder_adds_log_std():
    perturbed_y = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    t = np.array([[0.1], [0.5], [0.9]])
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=1.0)

    builder = RawTimeLogStdFeatureBuilder()
    features = builder.make_features(perturbed_y=perturbed_y, X=X, t=t, sde=sde)
    _, std = sde.get_mean_std_pt_given_y0(perturbed_y, t)
    expected = np.concatenate([perturbed_y, X, t, np.log(std[:, :1])], axis=1)

    assert builder.name == "raw_time_log_std"
    assert get_noise_feature_builder("raw_time_log_std").name == "raw_time_log_std"
    assert features.shape == expected.shape
    assert features.shape[1] == perturbed_y.shape[1] + X.shape[1] + 2
    assert np.allclose(features, expected)


def test_make_training_data_with_raw_time_preserves_shape_and_cat_idx():
    n, x_dim, y_dim = 30, 4, 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, x_dim))
    y = rng.normal(size=(n, y_dim))
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=1.0)
    cat_idx = [1, 3]
    n_repeats = 2
    eval_percent = 0.2

    baseline = _make_training_data(
        X=X,
        y=y,
        sde=sde,
        n_repeats=n_repeats,
        eval_percent=eval_percent,
        cat_idx=list(cat_idx),
        seed=0,
    )
    explicit = _make_training_data(
        X=X,
        y=y,
        sde=sde,
        n_repeats=n_repeats,
        eval_percent=eval_percent,
        cat_idx=list(cat_idx),
        seed=0,
        noise_feature_builder=RawTimeFeatureBuilder(),
    )

    pred_train_b, pred_val_b, target_train_b, target_val_b, cat_idx_b = baseline
    pred_train_e, pred_val_e, target_train_e, target_val_e, cat_idx_e = explicit

    assert np.array_equal(pred_train_b, pred_train_e)
    assert np.array_equal(pred_val_b, pred_val_e)
    assert np.array_equal(target_train_b, target_train_e)
    assert np.array_equal(target_val_b, target_val_e)

    assert pred_train_b.shape[1] == y_dim + x_dim + 1
    assert cat_idx_b == [c + y_dim for c in cat_idx]
    assert cat_idx_e == cat_idx_b


def test_make_training_data_with_raw_time_log_std_preserves_cat_idx():
    n, x_dim, y_dim = 30, 4, 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, x_dim))
    y = rng.normal(size=(n, y_dim))
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=1.0)
    cat_idx = [1, 3]

    predictors_train, predictors_val, target_train, target_val, transformed_cat_idx = _make_training_data(
        X=X,
        y=y,
        sde=sde,
        n_repeats=2,
        eval_percent=0.2,
        cat_idx=list(cat_idx),
        seed=0,
        noise_feature_builder=RawTimeLogStdFeatureBuilder(),
    )

    assert predictors_val is not None
    assert target_val is not None
    assert predictors_train.shape[1] == y_dim + x_dim + 2
    assert predictors_val.shape[1] == y_dim + x_dim + 2
    assert not np.array_equal(predictors_train[:, -1], predictors_train[:, -2])
    assert target_train.shape[1] == y_dim
    assert transformed_cat_idx == [c + y_dim for c in cat_idx]


def test_lightgbm_score_model_with_raw_time_matches_default():
    n, x_dim, y_dim = 200, 1, 1
    sigma = 0.00001
    X, y = generate_bimodal_linear_regression_data(n, x_dim, sigma, bimodal=False, seed=0)
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=float(y.std()))

    common = {
        "verbose": -1,
        "n_estimators": 20,
        "learning_rate": 0.1,
        "n_repeats": 1,
        "seed": 0,
    }
    default_model = LightGBMScoreModel(**common)
    explicit_model = LightGBMScoreModel(noise_features="raw_time", **common)

    default_model.fit(X, y, sde)
    explicit_model.fit(X, y, sde)

    rng = np.random.default_rng(0)
    random_t = rng.uniform(1e-5, sde.T / 2, size=n).reshape(-1, 1)
    z = rng.normal(size=(n, y_dim))
    mean, std = sde.get_mean_std_pt_given_y0(y, random_t)
    y_perturbed = mean + z * std

    scores_default = default_model.score(y=y_perturbed, X=X, t=random_t)
    scores_explicit = explicit_model.score(y=y_perturbed, X=X, t=random_t)

    assert np.allclose(scores_default, scores_explicit)


def test_lightgbm_score_model_with_raw_time_log_std_scores_finite_near_eps():
    n, x_dim = 160, 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, x_dim))
    y = np.column_stack(
        [
            X[:, 0] + rng.normal(scale=0.1, size=n),
            X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=n),
        ]
    )
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=float(y.std()))
    score_model = LightGBMScoreModel(
        noise_features="raw_time_log_std",
        verbose=-1,
        n_estimators=10,
        learning_rate=0.1,
        n_repeats=1,
        eval_percent=None,
        seed=0,
    )

    score_model.fit(X, y, sde)
    t = np.full((n, 1), 1e-5)
    z = rng.normal(size=y.shape)
    mean, std = sde.get_mean_std_pt_given_y0(y, t)
    y_perturbed = mean + z * std

    scores = score_model.score(y=y_perturbed, X=X, t=t)

    assert scores.shape == y.shape
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("noise_features", ["raw_time", "raw_time_log_std"])
def test_lightgbm_score_model_with_x0_scores_finite(noise_features):
    n, x_dim = 160, 2
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, x_dim))
    y = np.column_stack(
        [
            X[:, 0] + rng.normal(scale=0.1, size=n),
            X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=n),
        ]
    )
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=float(y.std()))
    score_model = LightGBMScoreModel(
        score_parameterization="x0",
        noise_features=noise_features,
        verbose=-1,
        n_estimators=10,
        learning_rate=0.1,
        n_repeats=1,
        eval_percent=None,
        seed=0,
    )

    score_model.fit(X, y, sde)
    t = np.concatenate([np.full((n // 2, 1), 1e-5), np.full((n - n // 2, 1), sde.T * 0.9)])
    z = rng.normal(size=y.shape)
    mean, std = sde.get_mean_std_pt_given_y0(y, t)
    y_perturbed = mean + z * std

    scores = score_model.score(y=y_perturbed, X=X, t=t)

    assert score_model.score_parameterization.name == "x0"
    assert scores.shape == y.shape
    assert np.all(np.isfinite(scores))


def test_linear_regression():
    """
    This test checks that the score model can fit a simple linear regression model.
    We do this by using the fact that for the VESDE model the score
    is -(y_perturbed - y_true)/sigma^2.  Hence

    Hence
        y_true = -score(y_perturbed; x, t) * sigma^2 + y_perturbed
    """

    # Params
    n = 1000
    x_dim = 1
    y_dim = 1
    sigma = 0.00001
    n_estimators = 100
    learning_rate = 0.01
    n_repeats = 10

    X, y = generate_bimodal_linear_regression_data(n, x_dim, sigma, bimodal=False, seed=0)

    assert X.shape == (n, x_dim)
    assert y.shape == (n, y_dim)

    # Fit a score model
    hyperparam_min = 0.01
    hyperparam_max = y.std()
    sde = VESDE(hyperparam_min=hyperparam_min, hyperparam_max=hyperparam_max)
    score_model = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
    )
    score_model.fit(X, y, sde)

    # Check that the score model is able to fit the data
    random_t = np.random.uniform(1e-5, sde.T // 2, size=n)
    random_t = repeat(random_t, "n -> n 1")
    z = np.random.randn(n)
    z = repeat(z, "n -> n y_dim", y_dim=y_dim)

    mean, std = sde.get_mean_std_pt_given_y0(y, random_t)
    y_perturbed = mean + z * std

    scores = score_model.score(y=y_perturbed, X=X, t=random_t)
    y_pred = (-1.0) * scores * sigma**2 + y_perturbed

    # Check that the R^2 is close to 1
    r2 = r2_score(y.flatten(), y_pred.flatten())
    assert r2 > 0.95, f"R^2 is {r2}"


def test_can_be_deterministic():
    # Params
    n = 200
    x_dim = 1
    y_dim = 1
    sigma = 0.00001
    n_estimators = 50
    learning_rate = 0.1
    n_repeats = 1

    X, y = generate_bimodal_linear_regression_data(n, x_dim, sigma, bimodal=False, seed=0)

    assert X.shape == (n, x_dim)
    assert y.shape == (n, y_dim)

    # Fit a score model
    hyperparam_min = 0.01
    hyperparam_max = y.std()
    sde = VESDE(hyperparam_min=hyperparam_min, hyperparam_max=hyperparam_max)
    seed = 0

    # First fit
    score_model_a = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed,
    )
    score_model_a.fit(X, y, sde)

    # Second fit
    score_model_b = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed,
    )
    score_model_b.fit(X, y, sde)

    # Check that the two results are the same
    random_t = np.random.uniform(1e-5, sde.T // 2, size=n)
    random_t = repeat(random_t, "n -> n 1")
    z = np.random.randn(n)
    z = repeat(z, "n -> n y_dim", y_dim=y_dim)

    mean, std = sde.get_mean_std_pt_given_y0(y, random_t)
    y_perturbed = mean + z * std

    scores_a = score_model_a.score(y=y_perturbed, X=X, t=random_t)
    scores_b = score_model_b.score(y=y_perturbed, X=X, t=random_t)

    msg = "The score model is not deterministic"
    assert np.allclose(scores_a, scores_b), msg


def test_different_seeds_do_not_give_same_results():
    # Params
    n = 200
    x_dim = 1
    y_dim = 1
    sigma = 0.00001
    n_estimators = 50
    learning_rate = 0.1
    n_repeats = 5

    X, y = generate_bimodal_linear_regression_data(n, x_dim, sigma, bimodal=False, seed=0)

    assert X.shape == (n, x_dim)
    assert y.shape == (n, y_dim)

    # Fit a score model
    hyperparam_min = 0.01
    hyperparam_max = y.std()
    sde = VESDE(hyperparam_min=hyperparam_min, hyperparam_max=hyperparam_max)

    seed_a = 0
    seed_b = 1

    # First fit
    score_model_a = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed_a,
    )
    score_model_a.fit(X, y, sde)

    # Second fit
    score_model_b = LightGBMScoreModel(
        verbose=1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_repeats=n_repeats,
        seed=seed_b,
    )
    score_model_b.fit(X, y, sde)

    # Check that the two results are the same
    random_t = np.random.uniform(1e-5, sde.T // 2, size=n)
    random_t = repeat(random_t, "n -> n 1")
    z = np.random.randn(n)
    z = repeat(z, "n -> n y_dim", y_dim=y_dim)

    mean, std = sde.get_mean_std_pt_given_y0(y, random_t)
    y_perturbed = mean + z * std

    scores_a = score_model_a.score(y=y_perturbed, X=X, t=random_t)
    scores_b = score_model_b.score(y=y_perturbed, X=X, t=random_t)

    # Check that the score model gives different results
    msg = "The score model gives the same results for different seeds"
    assert not np.allclose(scores_a, scores_b), msg


def test_make_training_data_respects_validation_split():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.zeros((20, 1))
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=1.0)

    predictors_train, predictors_val, _, _, _ = _make_training_data(
        X=X,
        y=y,
        sde=sde,
        n_repeats=3,
        eval_percent=0.25,
        seed=0,
    )

    x_train_ids = set(predictors_train[:, 1].astype(int))
    x_val_ids = set(predictors_val[:, 1].astype(int))
    assert x_train_ids.isdisjoint(x_val_ids)


def test_make_training_data_does_not_mutate_global_numpy_rng():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.zeros((10, 1))
    sde = VESDE(hyperparam_min=0.01, hyperparam_max=1.0)

    np.random.seed(123)
    expected = np.random.random(5)

    np.random.seed(123)
    _make_training_data(
        X=X,
        y=y,
        sde=sde,
        n_repeats=2,
        eval_percent=0.2,
        seed=999,
    )
    observed = np.random.random(5)

    assert np.allclose(observed, expected)
