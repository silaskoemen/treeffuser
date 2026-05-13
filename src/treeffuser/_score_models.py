"""
Contains different score models to be used to approximate the score of a given SDE.
"""

import abc
import warnings
from typing import Any
from typing import cast

import lightgbm as lgb
import numpy as np
from jaxtyping import Float
from jaxtyping import Int
from sklearn.model_selection import train_test_split

from treeffuser._flow_matching import FlowPath
from treeffuser._flow_matching import get_flow_path
from treeffuser.sde import DiffusionSDE

###################################################
# Score parameterizations
###################################################


class ScoreParameterization(abc.ABC):
    """
    Defines the regression target used to train a score model and how model predictions
    are converted back into a score during reverse-time sampling.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def make_target(
        self,
        y0: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        z: Float[np.ndarray, "batch y_dim"],
        mean: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        t: Float[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        pass

    @abc.abstractmethod
    def reconstruct_score(
        self,
        prediction: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        predicted_mean: Float[np.ndarray, "batch y_dim"] | None = None,
    ) -> Float[np.ndarray, "batch y_dim"]:
        pass

    @property
    def requires_prediction_mean(self) -> bool:
        return False

    def make_feature_perturbed_y(
        self,
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        return perturbed_y

    def make_prediction_y0(
        self,
        prediction: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        return prediction


class NoiseParameterization(ScoreParameterization):
    """
    Current Treeffuser behavior: train LightGBM to predict the added negative noise and
    reconstruct the score by dividing the prediction by the perturbation standard deviation.

    This corresponds to the denoising objective

        || std(t) * score(y_perturbed, x, t) - (mean(y0, t) - y_perturbed) / std(t) ||^2

    and, since y_perturbed = mean(y0, t) + std(t) * z, the fitted regression target is

        GBT(y_perturbed, x, t) = -z.
    """

    @property
    def name(self) -> str:
        return "noise"

    def make_target(
        self,
        y0: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        z: Float[np.ndarray, "batch y_dim"],
        mean: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        t: Float[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        return -1.0 * z

    def reconstruct_score(
        self,
        prediction: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        predicted_mean: Float[np.ndarray, "batch y_dim"] | None = None,
    ) -> Float[np.ndarray, "batch y_dim"]:
        return prediction / std


class X0Parameterization(ScoreParameterization):
    """
    Denoised target parameterization: train LightGBM to predict the clean response y0.

    Since the noising distribution is Gaussian,

        y_t | y0 ~ N(mean_t(y0), std(t)^2 I),

    the conditional score is

        score(y_t | y0, t) = (mean_t(y0) - y_t) / std(t)^2.

    The fitted model approximates E[y0 | y_t, x, t]. For the currently supported SDEs,
    mean_t(y0) is linear in y0, so plugging the denoised prediction into mean_t gives the
    corresponding marginal-score estimate.
    """

    @property
    def name(self) -> str:
        return "x0"

    def make_target(
        self,
        y0: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        z: Float[np.ndarray, "batch y_dim"],
        mean: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        t: Float[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        return y0

    def reconstruct_score(
        self,
        prediction: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        predicted_mean: Float[np.ndarray, "batch y_dim"] | None = None,
    ) -> Float[np.ndarray, "batch y_dim"]:
        if predicted_mean is None:
            raise ValueError("X0Parameterization requires mean_t(prediction) to reconstruct the score.")
        return (predicted_mean - perturbed_y) / (std**2)

    @property
    def requires_prediction_mean(self) -> bool:
        return True


class EDMParameterization(ScoreParameterization):
    """
    EDM-style preconditioned denoising parameterization.

    The coefficient formulas are the EDM preconditioning coefficients from
    Karras et al. For VESDE-style perturbations, `y_t = y0 + sigma * z`, they
    have the usual EDM variance-normalization interpretation. For VPSDE and
    SubVPSDE, this remains a valid preconditioned `x0` reparameterization, but
    the coefficients are no longer the Bayes-optimal skip/input normalizers.

    The noisy response is scaled before it is passed to the regressor,

        y_in = c_in(sigma) * y_t,

    and the regressor target is the preconditioned residual needed by the EDM
    denoiser

        D(y_t, sigma) = c_skip(sigma) * y_t + c_out(sigma) * F(y_in, x, sigma).

    With standardized targets, `sigma_data=1` is the natural default. Treeffuser
    standardizes `y` before score-model fitting, so this default matches the
    public estimator path. Training the regressor on the residual target is
    equivalent to the EDM weighted denoising objective because the usual EDM loss
    weight cancels `c_out`.
    """

    def __init__(self, sigma_data: float = 1.0) -> None:
        if sigma_data <= 0:
            raise ValueError("sigma_data must be strictly positive.")
        self.sigma_data = sigma_data

    @property
    def name(self) -> str:
        return "edm"

    def make_target(
        self,
        y0: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        z: Float[np.ndarray, "batch y_dim"],
        mean: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        t: Float[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        c_skip, c_out, _ = self._coefficients(std)
        return (y0 - c_skip * perturbed_y) / c_out

    def reconstruct_score(
        self,
        prediction: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        predicted_mean: Float[np.ndarray, "batch y_dim"] | None = None,
    ) -> Float[np.ndarray, "batch y_dim"]:
        if predicted_mean is None:
            raise ValueError("EDMParameterization requires mean_t(D(y_t, sigma)) to reconstruct the score.")
        # `prediction` has already been converted into the denoised mean path.
        return (predicted_mean - perturbed_y) / (std**2)

    @property
    def requires_prediction_mean(self) -> bool:
        return True

    def make_feature_perturbed_y(
        self,
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        _, _, c_in = self._coefficients(std)
        return c_in * perturbed_y

    def make_prediction_y0(
        self,
        prediction: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        c_skip, c_out, _ = self._coefficients(std)
        return c_skip * perturbed_y + c_out * prediction

    def _coefficients(
        self,
        std: Float[np.ndarray, "batch y_dim"],
    ) -> tuple[
        Float[np.ndarray, "batch y_dim"],
        Float[np.ndarray, "batch y_dim"],
        Float[np.ndarray, "batch y_dim"],
    ]:
        if np.any(std <= 0):
            raise ValueError("EDMParameterization requires strictly positive SDE std values.")
        sigma_data_sq = self.sigma_data**2
        denom = std**2 + sigma_data_sq
        c_skip = sigma_data_sq / denom
        c_out = std * self.sigma_data / np.sqrt(denom)
        c_in = 1.0 / np.sqrt(denom)
        return c_skip, c_out, c_in


def get_score_parameterization(
    parameterization: str | ScoreParameterization,
    edm_sigma_data: float = 1.0,
) -> ScoreParameterization:
    if isinstance(parameterization, ScoreParameterization):
        return parameterization
    if parameterization == "noise":
        return NoiseParameterization()
    if parameterization == "x0":
        return X0Parameterization()
    if parameterization == "edm":
        return EDMParameterization(sigma_data=edm_sigma_data)
    raise ValueError(f"Unknown score parameterization: {parameterization}")


###################################################
# Noise feature builders
###################################################


class NoiseFeatureBuilder(abc.ABC):
    """
    Builds the feature matrix passed to the underlying regressor from the perturbed
    sample, side information, and time. Centralizing this lets training and inference
    share one definition so the two paths cannot drift apart.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def make_features(
        self,
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Float[np.ndarray, "batch 1"],
        sde: DiffusionSDE,
        std: Float[np.ndarray, "batch y_dim"] | None = None,
    ) -> Float[np.ndarray, "batch feat_dim"]:
        pass


class RawTimeFeatureBuilder(NoiseFeatureBuilder):
    """
    Current Treeffuser feature layout: [perturbed_y, X, t].
    """

    @property
    def name(self) -> str:
        return "raw_time"

    def make_features(
        self,
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Float[np.ndarray, "batch 1"],
        sde: DiffusionSDE,
        std: Float[np.ndarray, "batch y_dim"] | None = None,
    ) -> Float[np.ndarray, "batch feat_dim"]:
        return np.concatenate([perturbed_y, X, t], axis=1)


class RawTimeLogStdFeatureBuilder(NoiseFeatureBuilder):
    """
    Treeffuser feature layout with explicit noise scale: [perturbed_y, X, t, log_std(t)].
    """

    @property
    def name(self) -> str:
        return "raw_time_log_std"

    def make_features(
        self,
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Float[np.ndarray, "batch 1"],
        sde: DiffusionSDE,
        std: Float[np.ndarray, "batch y_dim"] | None = None,
    ) -> Float[np.ndarray, "batch feat_dim"]:
        if std is None:
            _, std = sde.get_mean_std_pt_given_y0(perturbed_y, t)
        std_col = std[:, :1]
        if not np.allclose(std, std_col):
            raise ValueError("raw_time_log_std requires the SDE std to be identical across y dimensions.")
        if np.any(std_col <= 0):
            raise ValueError("raw_time_log_std requires strictly positive SDE std values.")
        log_std = np.log(std_col)
        return np.concatenate([perturbed_y, X, t, log_std], axis=1)


def get_noise_feature_builder(
    feature_builder: str | NoiseFeatureBuilder,
) -> NoiseFeatureBuilder:
    if isinstance(feature_builder, NoiseFeatureBuilder):
        return feature_builder
    if feature_builder == "raw_time":
        return RawTimeFeatureBuilder()
    if feature_builder == "raw_time_log_std":
        return RawTimeLogStdFeatureBuilder()
    raise ValueError(f"Unknown noise feature builder: {feature_builder}")


###################################################
# Loss weighting
###################################################


class LossWeighting(abc.ABC):
    """
    Per-sample loss weighting for the score-model regression. Implementations return a
    1-D array of weights (length batch) that is multiplied onto LightGBM's squared loss.

    The mapping from a target denoising-score-matching (DSM) weight schedule to a
    sample weight on the regressor's target depends on the score parameterization,
    so implementations are passed the active `ScoreParameterization` and may dispatch
    accordingly.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def compute_sample_weight(
        self,
        std: Float[np.ndarray, "batch y_dim"],
        alpha: Float[np.ndarray, "batch y_dim"],
        parameterization: ScoreParameterization,
    ) -> Float[np.ndarray, "batch"]:
        pass


class UniformLossWeighting(LossWeighting):
    """
    Sample weight 1 for every row. Reproduces the current Treeffuser behavior.
    """

    @property
    def name(self) -> str:
        return "uniform"

    def compute_sample_weight(
        self,
        std: Float[np.ndarray, "batch y_dim"],
        alpha: Float[np.ndarray, "batch y_dim"],
        parameterization: ScoreParameterization,
    ) -> Float[np.ndarray, "batch"]:
        return np.ones(std.shape[0], dtype=np.float64)


class MinSNRLossWeighting(LossWeighting):
    """
    Min-SNR-gamma loss weighting from Hang et al. (2023), "Efficient Diffusion Training
    via Min-SNR Weighting Strategy".

    SNR(t) = alpha(t)^2 / std(t)^2. The sample weight on the regression target is
    parameterization-aware so the implied DSM loss weight is `min(SNR, gamma)`:

        - NoiseParameterization (target = -z): w = min(SNR, gamma) / SNR
        - X0Parameterization   (target =  y0): w = min(SNR, gamma)
        - EDMParameterization  (preconditioned residual target): w = min(SNR, gamma) * c_out(sigma)^2

    Clipping at `gamma` caps the influence of small-noise rows where the score is
    unboundedly large, while keeping pressure on the regime that drives local density
    shape and interval calibration.
    """

    def __init__(self, gamma: float = 5.0) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be strictly positive.")
        self.gamma = float(gamma)

    @property
    def name(self) -> str:
        return f"min_snr_g{self.gamma:g}"

    def compute_sample_weight(
        self,
        std: Float[np.ndarray, "batch y_dim"],
        alpha: Float[np.ndarray, "batch y_dim"],
        parameterization: ScoreParameterization,
    ) -> Float[np.ndarray, "batch"]:
        if np.any(std <= 0):
            raise ValueError("MinSNRLossWeighting requires strictly positive std values.")
        std_col = std[:, 0]
        alpha_col = alpha[:, 0]
        snr = (alpha_col / std_col) ** 2
        clipped = np.minimum(snr, self.gamma)
        if isinstance(parameterization, NoiseParameterization):
            return clipped / snr
        if isinstance(parameterization, X0Parameterization):
            return clipped
        if isinstance(parameterization, EDMParameterization):
            sigma_data_sq = parameterization.sigma_data**2
            c_out_sq = (std_col**2) * sigma_data_sq / (std_col**2 + sigma_data_sq)
            return clipped * c_out_sq
        raise NotImplementedError(
            f"MinSNRLossWeighting is not implemented for parameterization " f"{type(parameterization).__name__}."
        )


def get_loss_weighting(
    spec: str | LossWeighting,
    min_snr_gamma: float = 5.0,
) -> LossWeighting:
    if isinstance(spec, LossWeighting):
        return spec
    if spec == "uniform":
        return UniformLossWeighting()
    if spec == "min_snr":
        return MinSNRLossWeighting(gamma=min_snr_gamma)
    raise ValueError(f"Unknown loss weighting: {spec!r}")


###################################################
# t sampling
###################################################

# Smallest t we ever sample. Matches the EPS used historically in `_make_training_data`
# and the reverse-time integration endpoint in `_base_tabular_diffusion.sample`.
_T_SAMPLER_EPS = 1e-5
_FLOW_MATCHING_T_EPS = _T_SAMPLER_EPS
_FLOW_MATCHING_ENDPOINT_FRACTION = 0.05


class TSampler(abc.ABC):
    """
    Draws training `t` values from a chosen distribution.

    Tree-based score models bin features by data density: shifting the distribution of
    `t` values changes where LightGBM places its histogram splits and therefore where
    the score model spends capacity. This is a different lever from loss weighting,
    which scales the loss within fixed bins.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def sample(
        self,
        n: int,
        sde: DiffusionSDE,
        rng: np.random.Generator,
    ) -> Float[np.ndarray, "n 1"]:
        pass


class UniformTSampler(TSampler):
    """Uniform `t` on `[EPS, T]`. Reproduces the current Treeffuser behavior."""

    @property
    def name(self) -> str:
        return "uniform"

    def sample(
        self,
        n: int,
        sde: DiffusionSDE,
        rng: np.random.Generator,
    ) -> Float[np.ndarray, "n 1"]:
        return rng.uniform(0, 1, size=(n, 1)) * (sde.T - _T_SAMPLER_EPS) + _T_SAMPLER_EPS


class LogSigmaNormalTSampler(TSampler):
    """
    EDM-style `t` sampling: draws `log sigma(t) ~ Normal(p_mean, p_std)`, clips to the
    SDE's achievable `[log sigma(EPS), log sigma(T)]` range, then inverts back to `t`
    using a precomputed `t -> sigma` lookup with `np.interp`. The lookup is rebuilt on
    each call so any SDE hyperparameter changes between fits are picked up.

    Parameters
    ----------
    p_mean, p_std : float
        Mean and standard deviation of the log-sigma Normal. EDM uses (-1.2, 1.2) for
        image data on standardized targets; Treeffuser uses standardized targets too so
        the defaults are a reasonable starting point.
    table_size : int
        Number of `t` grid points used for the lookup. 1024 is overkill for VESDE
        (analytic geometric schedule) and adequate for VPSDE/SubVPSDE.
    """

    def __init__(self, p_mean: float = -1.2, p_std: float = 1.2, table_size: int = 1024) -> None:
        if p_std <= 0:
            raise ValueError("p_std must be strictly positive.")
        if table_size < 16:
            raise ValueError("table_size must be at least 16.")
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.table_size = int(table_size)

    @property
    def name(self) -> str:
        return f"log_sigma_normal_pm{self.p_mean:g}_ps{self.p_std:g}"

    def _build_table(self, sde: DiffusionSDE) -> tuple[np.ndarray, np.ndarray]:
        t_grid = np.linspace(_T_SAMPLER_EPS, sde.T, self.table_size).reshape(-1, 1)
        _, std_grid = sde.get_mean_std_pt_given_y0(np.ones((self.table_size, 1)), t_grid)
        std_col = std_grid[:, 0]
        if np.any(std_col <= 0):
            raise ValueError(
                "LogSigmaNormalTSampler requires the SDE to have strictly positive " "std(t) on the sampling interval."
            )
        log_sigma_grid = np.log(std_col)
        # The SDE schedules supported here are monotone in t. If not strictly increasing
        # the interpolation behaves like nearest-neighbor on flats; we sort defensively.
        order = np.argsort(log_sigma_grid)
        return log_sigma_grid[order], t_grid[:, 0][order]

    def sample(
        self,
        n: int,
        sde: DiffusionSDE,
        rng: np.random.Generator,
    ) -> Float[np.ndarray, "n 1"]:
        log_sigma_grid, t_grid = self._build_table(sde)
        log_sigma = rng.normal(loc=self.p_mean, scale=self.p_std, size=n)
        log_sigma = np.clip(log_sigma, log_sigma_grid[0], log_sigma_grid[-1])
        t = np.interp(log_sigma, log_sigma_grid, t_grid)
        return t.reshape(-1, 1)


def get_t_sampler(
    spec: str | TSampler,
    log_sigma_p_mean: float = -1.2,
    log_sigma_p_std: float = 1.2,
) -> TSampler:
    if isinstance(spec, TSampler):
        return spec
    if spec == "uniform":
        return UniformTSampler()
    if spec == "log_sigma_normal":
        return LogSigmaNormalTSampler(p_mean=log_sigma_p_mean, p_std=log_sigma_p_std)
    raise ValueError(f"Unknown t sampling strategy: {spec!r}")


###################################################
# Helper functions
###################################################


def _fit_one_lgbm_model(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    X_val: Float[np.ndarray, "batch x_dim"] | None,
    y_val: Float[np.ndarray, "batch y_dim"] | None,
    seed: int | None,
    verbose: int,
    cat_idx: list[int] | None = None,
    n_jobs: int | None = -1,
    early_stopping_rounds: int | None = None,
    sample_weight: Float[np.ndarray, "batch"] | None = None,
    sample_weight_val: Float[np.ndarray, "batch"] | None = None,
    **lgbm_args,
) -> lgb.LGBMRegressor:
    """
    Simple wrapper for fitting a lightgbm model. See
    the lightgbm score function documentation for more details.
    """
    callbacks = None
    if early_stopping_rounds is not None:
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose > 0)]

    model = lgb.LGBMRegressor(
        random_state=seed,
        verbose=verbose,
        n_jobs=n_jobs,
        linear_tree=False,
        **lgbm_args,
    )
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        eval_sample_weight = [sample_weight_val] if sample_weight_val is not None else None
    else:
        eval_set = None
        eval_sample_weight = None
    categorical_feature: list[int] | str = "auto" if cat_idx is None else cat_idx
    model.fit(
        X=X,
        y=y,
        sample_weight=sample_weight,
        eval_set=cast(Any, eval_set),
        eval_sample_weight=cast(Any, eval_sample_weight),
        callbacks=cast(Any, callbacks),
        categorical_feature=categorical_feature,
    )
    return model


def _make_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    sde: DiffusionSDE,
    n_repeats: int | None,
    eval_percent: float | None,
    cat_idx: list[int] | None = None,
    seed: int | None = None,
    score_parameterization: ScoreParameterization | None = None,
    noise_feature_builder: NoiseFeatureBuilder | None = None,
    loss_weighting: LossWeighting | None = None,
    t_sampler: TSampler | None = None,
):
    """
    Creates the training data for the score model. The score parameterization owns the
    regression target; the noise feature builder owns the LightGBM feature matrix layout;
    the loss weighting owns the per-sample regression weights.

    Returns:
    - predictors_train: training features for lgbm
    - predictors_val: validation features for lgbm
    - predicted_train: training target for lgbm
    - predicted_val: validation target for lgbm
    - sample_weight_train: per-row weights for the training MSE (or `None` if uniform)
    - sample_weight_val: per-row weights for the validation MSE (or `None`)
    - cat_idx: shifted categorical-feature indices
    """
    if score_parameterization is None:
        score_parameterization = NoiseParameterization()
    if noise_feature_builder is None:
        noise_feature_builder = RawTimeFeatureBuilder()
    if loss_weighting is None:
        loss_weighting = UniformLossWeighting()
    if t_sampler is None:
        t_sampler = UniformTSampler()
    rng = np.random.default_rng(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    predictors_train, predictors_val = None, None
    predicted_train, predicted_val = None, None
    sample_weight_train: Float[np.ndarray, "batch"] | None = None
    sample_weight_val: Float[np.ndarray, "batch"] | None = None

    if eval_percent is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=eval_percent, random_state=seed)

    # TRAINING DATA
    n_reps = n_repeats if n_repeats is not None else 1
    X_train = np.tile(X_train, (n_reps, 1))
    y_train = np.tile(y_train, (n_reps, 1))
    t_train = t_sampler.sample(y_train.shape[0], sde=sde, rng=rng)
    z_train = rng.normal(size=y_train.shape)

    train_mean, train_std = sde.get_mean_std_pt_given_y0(y_train, t_train)
    train_alpha, _ = sde.get_mean_std_pt_given_y0(np.ones_like(y_train), t_train)
    perturbed_y_train = train_mean + train_std * z_train
    feature_perturbed_y_train = score_parameterization.make_feature_perturbed_y(
        perturbed_y=perturbed_y_train,
        std=train_std,
    )
    predictors_train = noise_feature_builder.make_features(
        perturbed_y=feature_perturbed_y_train,
        X=X_train,
        t=t_train,
        sde=sde,
        std=train_std,
    )
    predicted_train = score_parameterization.make_target(
        y0=y_train,
        perturbed_y=perturbed_y_train,
        z=z_train,
        mean=train_mean,
        std=train_std,
        t=t_train,
    )
    if not isinstance(loss_weighting, UniformLossWeighting):
        sample_weight_train = loss_weighting.compute_sample_weight(
            std=train_std,
            alpha=train_alpha,
            parameterization=score_parameterization,
        )

    # VALIDATION DATA
    if eval_percent is not None:
        assert y_test is not None
        assert X_test is not None
        t_val = t_sampler.sample(y_test.shape[0], sde=sde, rng=rng)
        z_val = rng.normal(size=(y_test.shape[0], y_test.shape[1]))

        val_mean, val_std = sde.get_mean_std_pt_given_y0(y_test, t_val)
        val_alpha, _ = sde.get_mean_std_pt_given_y0(np.ones_like(y_test), t_val)
        perturbed_y_val = val_mean + val_std * z_val
        feature_perturbed_y_val = score_parameterization.make_feature_perturbed_y(
            perturbed_y=perturbed_y_val,
            std=val_std,
        )
        predictors_val = noise_feature_builder.make_features(
            perturbed_y=feature_perturbed_y_val,
            X=X_test,
            t=t_val,
            sde=sde,
            std=val_std,
        )
        predicted_val = score_parameterization.make_target(
            y0=y_test,
            perturbed_y=perturbed_y_val,
            z=z_val,
            mean=val_mean,
            std=val_std,
            t=t_val,
        )
        if not isinstance(loss_weighting, UniformLossWeighting):
            sample_weight_val = loss_weighting.compute_sample_weight(
                std=val_std,
                alpha=val_alpha,
                parameterization=score_parameterization,
            )

    cat_idx = [c + y_train.shape[1] for c in cat_idx] if cat_idx is not None else None

    return (
        predictors_train,
        predictors_val,
        predicted_train,
        predicted_val,
        sample_weight_train,
        sample_weight_val,
        cat_idx,
    )


def _sample_flow_matching_t(n: int, rng: np.random.Generator) -> Float[np.ndarray, "n 1"]:
    t = rng.uniform(_FLOW_MATCHING_T_EPS, 1.0, size=(n, 1))
    endpoint_count = max(1, round(n * _FLOW_MATCHING_ENDPOINT_FRACTION)) if n > 0 else 0
    if endpoint_count > 0:
        endpoint_idx = rng.choice(n, size=endpoint_count, replace=False)
        t[endpoint_idx] = 1.0
    return t


def _make_flow_matching_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    flow_path: FlowPath,
    n_repeats: int | None,
    eval_percent: float | None,
    cat_idx: list[int] | None = None,
    seed: int | None = None,
    noise_feature_builder: NoiseFeatureBuilder | None = None,
):
    """
    Creates LightGBM training rows for linear flow matching.

    The validation split is made on original `(X, y0)` rows before repeats and
    prior-noise draws, matching `_make_training_data` and avoiding leakage between
    noisy views of the same data point.
    """
    if noise_feature_builder is None:
        noise_feature_builder = RawTimeFeatureBuilder()
    rng = np.random.default_rng(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    predictors_val = None
    predicted_val = None

    if eval_percent is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=eval_percent, random_state=seed)

    n_reps = n_repeats if n_repeats is not None else 1
    X_train = np.tile(X_train, (n_reps, 1))
    y_train = np.tile(y_train, (n_reps, 1))
    t_train = _sample_flow_matching_t(y_train.shape[0], rng)
    z_train = flow_path.sample_prior(y_train.shape, rng=rng)
    y_t_train = flow_path.interpolate(y0=y_train, z=z_train, t=t_train)
    predictors_train = noise_feature_builder.make_features(
        perturbed_y=y_t_train,
        X=X_train,
        t=t_train,
        sde=cast(DiffusionSDE, None),
    )
    predicted_train = flow_path.target_velocity(y0=y_train, z=z_train, t=t_train)

    if eval_percent is not None:
        assert y_test is not None
        assert X_test is not None
        t_val = _sample_flow_matching_t(y_test.shape[0], rng)
        z_val = flow_path.sample_prior(y_test.shape, rng=rng)
        y_t_val = flow_path.interpolate(y0=y_test, z=z_val, t=t_val)
        predictors_val = noise_feature_builder.make_features(
            perturbed_y=y_t_val,
            X=X_test,
            t=t_val,
            sde=cast(DiffusionSDE, None),
        )
        predicted_val = flow_path.target_velocity(y0=y_test, z=z_val, t=t_val)

    cat_idx = [c + y_train.shape[1] for c in cat_idx] if cat_idx is not None else None

    return (
        predictors_train,
        predictors_val,
        predicted_train,
        predicted_val,
        cat_idx,
    )


###################################################
# Score models
###################################################


class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ):

        pass

    @abc.abstractmethod
    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: list[int] | None = None,
    ):
        pass


class VelocityModel(abc.ABC):
    flow_path: FlowPath

    @abc.abstractmethod
    def sample_prior(
        self,
        shape: tuple[int, ...],
        seed: int | None = None,
    ) -> Float[np.ndarray, "*shape"]:
        pass

    @abc.abstractmethod
    def velocity(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        pass

    @abc.abstractmethod
    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        cat_idx: list[int] | None = None,
    ):
        pass


class LightGBMScoreModel(ScoreModel):
    """
    A score model that uses a LightGBM model (trees) to approximate the score of a given SDE.

    Parameters
    ----------
    n_repeats : int
        How many times to repeat the training dataset when fitting the score. That is, how many
        noisy versions of a point to generate for training.
    eval_percent : float
        Percentage of the training data to use for validation for optional early stopping. It is
        ignored if `early_stopping_rounds` is not set in the `lgbm_args`.
    n_jobs : int
        LightGBM: Number of parallel threads. If set to -1, the number is set to the number of available cores.
    seed : int
        Random seed for generating the training data and fitting the model.
    verbose : int
        Verbosity of the score model.
    **lgbm_args
        Additional arguments to pass to the LightGBM model. See the LightGBM documentation for more
        information. E.g. `early_stopping_rounds`, `n_estimators`, `learning_rate`, `max_depth`,

    Attributes
    ----------
    n_estimators_true : List[int]
        The true number of trees in each model (in case early stopping is used).
    """

    def __init__(
        self,
        n_repeats: int | None = 10,
        eval_percent: float = 0.1,
        n_jobs: int | None = -1,
        seed: int | None = None,
        score_parameterization: str | ScoreParameterization = "noise",
        noise_features: str | NoiseFeatureBuilder = "raw_time",
        edm_sigma_data: float = 1.0,
        loss_weighting: str | LossWeighting = "uniform",
        min_snr_gamma: float = 5.0,
        t_sampling: str | TSampler = "uniform",
        log_sigma_p_mean: float = -1.2,
        log_sigma_p_std: float = 1.2,
        **lgbm_args,
    ) -> None:
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.n_jobs = n_jobs
        self.seed = seed
        self.score_parameterization = get_score_parameterization(
            score_parameterization,
            edm_sigma_data=edm_sigma_data,
        )
        self.noise_feature_builder = get_noise_feature_builder(noise_features)
        self.loss_weighting = get_loss_weighting(loss_weighting, min_snr_gamma=min_snr_gamma)
        self.t_sampler = get_t_sampler(
            t_sampling,
            log_sigma_p_mean=log_sigma_p_mean,
            log_sigma_p_std=log_sigma_p_std,
        )

        self._lgbm_args = lgbm_args
        self.sde = None
        self.models = None  # Convention inputs are (y, x, t)
        self.n_estimators_true = None

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        if self.sde is None:
            raise ValueError("The model has not been fitted yet.")
        assert self.models is not None

        predictions = []
        _, std = self.sde.get_mean_std_pt_given_y0(y, t)
        feature_perturbed_y = self.score_parameterization.make_feature_perturbed_y(
            perturbed_y=y,
            std=std,
        )
        predictors = self.noise_feature_builder.make_features(
            perturbed_y=feature_perturbed_y,
            X=X,
            t=t,
            sde=self.sde,
            std=std,
        )
        for i in range(y.shape[-1]):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                prediction_i = self.models[i].predict(predictors, num_threads=self.n_jobs)
            predictions.append(prediction_i)
        predictions = np.array(predictions).T
        predicted_mean = None
        if self.score_parameterization.requires_prediction_mean:
            prediction_y0 = self.score_parameterization.make_prediction_y0(
                prediction=predictions,
                perturbed_y=y,
                std=std,
            )
            predicted_mean, _ = self.sde.get_mean_std_pt_given_y0(prediction_y0, t)
        return self.score_parameterization.reconstruct_score(
            prediction=predictions,
            perturbed_y=y,
            std=std,
            predicted_mean=predicted_mean,
        )

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: list[int] | None = None,
    ):
        """
        Fit the score model to the data and the given SDE.

        Parameters
        ----------
        X : Float[np.ndarray, "batch x_dim"]
            The input data.
        y : Float[np.ndarray, "batch y_dim"]
            The true output values.
        sde : DiffusionSDE
            The SDE that the model is supposed to approximate the score of.
        cat_idx : Optional[List[int]]
            List of indices of categorical features in the input data. If `None`, all features are
            assumed to be continuous.
        """
        y_dim = y.shape[1]
        self.sde = sde
        self._warn_on_edm_config(sde)

        (
            lgb_X_train,
            lgb_X_val,
            lgb_y_train,
            lgb_y_val,
            sample_weight_train,
            sample_weight_val,
            cat_idx,
        ) = _make_training_data(
            X=X,
            y=y,
            sde=sde,
            n_repeats=self.n_repeats,
            eval_percent=self.eval_percent,
            cat_idx=cat_idx,
            seed=self.seed,
            score_parameterization=self.score_parameterization,
            noise_feature_builder=self.noise_feature_builder,
            loss_weighting=self.loss_weighting,
            t_sampler=self.t_sampler,
        )

        models = []
        for i in range(y_dim):
            lgb_y_val_i = lgb_y_val[:, i] if lgb_y_val is not None else None
            score_model_i = _fit_one_lgbm_model(
                X=lgb_X_train,
                y=lgb_y_train[:, i],
                X_val=lgb_X_val,
                y_val=lgb_y_val_i,
                cat_idx=cat_idx,
                seed=self.seed,
                n_jobs=self.n_jobs,
                sample_weight=sample_weight_train,
                sample_weight_val=sample_weight_val,
                **self._lgbm_args,
            )
            models.append(score_model_i)
        self.models = models

        # collect the true number of trees learned by each model
        self.n_estimators_true = [model.n_estimators_ for model in self.models]

    def _warn_on_edm_config(self, sde: DiffusionSDE) -> None:
        if not isinstance(self.score_parameterization, EDMParameterization):
            return
        if sde.__class__.__name__ != "VESDE":
            warnings.warn(
                "score_parameterization='edm' is an EDM-style x0 reparameterization for "
                "non-VESDE SDEs; the EDM input/skip coefficients are not Bayes-optimal "
                "for VPSDE/SubVPSDE marginals.",
                UserWarning,
                stacklevel=2,
            )
        if self.noise_feature_builder.name == "raw_time":
            warnings.warn(
                "score_parameterization='edm' is best paired with noise_features='raw_time_log_std' "
                "so the regressor receives an explicit log-noise feature.",
                UserWarning,
                stacklevel=2,
            )


class LightGBMVelocityModel(VelocityModel):
    """
    A LightGBM model that approximates a flow-matching velocity field.

    The learned object is `velocity(y_t, X, t)`, not a score. Sampling should use a
    deterministic reverse-velocity ODE rather than the reverse-SDE/PF-ODE wrappers.
    """

    def __init__(
        self,
        n_repeats: int | None = 10,
        eval_percent: float = 0.1,
        n_jobs: int | None = -1,
        seed: int | None = None,
        flow_path: str | FlowPath = "linear",
        noise_features: str | NoiseFeatureBuilder = "raw_time",
        verbose: int = 0,
        **lgbm_args,
    ) -> None:
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.n_jobs = n_jobs
        self.seed = seed
        self.flow_path = get_flow_path(flow_path)
        self.noise_feature_builder = get_noise_feature_builder(noise_features)
        if not isinstance(self.noise_feature_builder, RawTimeFeatureBuilder):
            raise ValueError("Flow matching currently supports only noise_features='raw_time'.")
        self.verbose = verbose
        self._lgbm_args = lgbm_args
        self.models = None
        self.n_estimators_true = None

    def sample_prior(
        self,
        shape: tuple[int, ...],
        seed: int | None = None,
    ) -> Float[np.ndarray, "*shape"]:
        return self.flow_path.sample_prior(shape, seed=seed)

    def velocity(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        if self.models is None:
            raise ValueError("The model has not been fitted yet.")

        predictors = self.noise_feature_builder.make_features(
            perturbed_y=y,
            X=X,
            t=t,
            sde=cast(DiffusionSDE, None),
        )
        predictions = []
        for i in range(y.shape[-1]):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                prediction_i = self.models[i].predict(predictors, num_threads=self.n_jobs)
            predictions.append(prediction_i)
        return np.array(predictions).T

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        cat_idx: list[int] | None = None,
    ):
        y_dim = y.shape[1]
        (
            lgb_X_train,
            lgb_X_val,
            lgb_y_train,
            lgb_y_val,
            cat_idx,
        ) = _make_flow_matching_training_data(
            X=X,
            y=y,
            flow_path=self.flow_path,
            n_repeats=self.n_repeats,
            eval_percent=self.eval_percent,
            cat_idx=cat_idx,
            seed=self.seed,
            noise_feature_builder=self.noise_feature_builder,
        )

        models = []
        for i in range(y_dim):
            lgb_y_val_i = lgb_y_val[:, i] if lgb_y_val is not None else None
            velocity_model_i = _fit_one_lgbm_model(
                X=lgb_X_train,
                y=lgb_y_train[:, i],
                X_val=lgb_X_val,
                y_val=lgb_y_val_i,
                cat_idx=cat_idx,
                seed=self.seed,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                **self._lgbm_args,
            )
            models.append(velocity_model_i)
        self.models = models
        self.n_estimators_true = [model.n_estimators_ for model in self.models]
