from __future__ import annotations

import warnings
from typing import Literal

from treeffuser._base_tabular_diffusion import BaseTabularDiffusion
from treeffuser._residualizer import ResidualizeMode
from treeffuser._score_models import LightGBMScoreModel
from treeffuser._score_models import LightGBMVelocityModel
from treeffuser._score_models import ScoreModel
from treeffuser._score_models import VelocityModel
from treeffuser.sde import DiffusionSDE
from treeffuser.sde import get_diffusion_sde


class Treeffuser(BaseTabularDiffusion):
    def __init__(
        self,
        n_repeats: int = 30,
        n_estimators: int = 3000,
        early_stopping_rounds: int | None = 50,
        eval_percent: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        max_bin: int = 255,
        subsample_for_bin: int = 200000,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        n_jobs: int = -1,
        sde_name: str = "vesde",
        sde_initialize_from_data: bool = False,
        sde_hyperparam_min: float | Literal["default"] | None = None,
        sde_hyperparam_max: float | Literal["default"] | None = None,
        training_objective: Literal["score", "flow_matching"] = "score",
        flow_path: str = "linear",
        score_parameterization: str = "noise",
        noise_features: str = "raw_time",
        edm_sigma_data: float = 1.0,
        loss_weighting: str = "uniform",
        min_snr_gamma: float = 5.0,
        t_sampling: str = "uniform",
        uniform_endpoint_fraction: float = 0.05,
        log_sigma_p_mean: float = -1.2,
        log_sigma_p_std: float = 1.2,
        residualize: ResidualizeMode = "off",
        residualize_k_folds: int = 5,
        seed: int | None = None,
        verbose: int = 0,
        extra_lightgbm_params: dict | None = None,
        extra_residualizer_params: dict | None = None,
    ):
        """
        n_repeats : int
            How many times to repeat the training dataset when fitting the score. That is, how many
            noisy versions of a point to generate for training.
        n_estimators : int
            LightGBM: Number of boosting iterations.
        early_stopping_rounds : int
            LightGBM: If `None`, no early stopping is performed. Otherwise, the model will stop training
            if no improvement is observed in the validation set for `early_stopping_rounds` consecutive
            iterations.
        eval_percent : float
            LightGBM: Percentage of the training data to use for validation if `early_stopping_rounds`
            is not `None`.
        num_leaves : int
            LightGBM: Maximum tree leaves for base learners.
        max_depth : int
            LightGBM: Maximum tree depth for base learners, <=0 means no limit.
        learning_rate : float
            LightGBM: Boosting learning rate.
        max_bin : int
            LightGBM: Max number of bins that feature values will be bucketed in. This is used for
            lightgbm's histogram binning algorithm.
        subsample_for_bin : int
            LightGBM: Number of samples for constructing bins.
        min_child_samples : int
            LightGBM: Minimum number of data needed in a child (leaf). If less than this number, will
            not create the child.
        subsample : float
            LightGBM: Subsample ratio of the training instance.
        subsample_freq : int
            LightGBM: Frequency of subsample, <=0 means no enable. How often to subsample the training
            data.
        n_jobs : int
            LightGBM: Number of parallel threads. If set to -1, the number is set to the number of available cores.
        sde_name : str
            SDE: Name of the SDE to use. See `treeffuser.sde.get_diffusion_sde` for available SDEs.
        sde_initialize_from_data : bool
            SDE: Whether to initialize the SDE from the data. If `True`, the SDE hyperparameters are
            initialized with a heuristic based on the data (see `treeffuser.sde.initialize.py`).
            Otherwise, sde_hyperparam_min and sde_hyperparam_max are used. (default: False)
        sde_hyperparam_min : float or "default"
            SDE: The scale of the SDE at t=0 (see `VESDE`, `VPSDE`, `SubVPSDE`).
        sde_hyperparam_max : float or "default"
            SDE: The scale of the SDE at t=T (see `VESDE`, `VPSDE`, `SubVPSDE`).
        training_objective : {"score", "flow_matching"}
            Objective used to train the generative model. "score" preserves the existing
            score-SDE behavior. "flow_matching" trains a direct velocity field and uses
            deterministic reverse-velocity ODE sampling.
        flow_path : {"linear", "trig", "vp"}
            Probability path used when `training_objective="flow_matching"`. "linear" is
            the rectified-flow path `y_t = (1-t) y0 + t z`; "trig" is the variance-
            preserving cosine/sine path; "vp" is the DDPM-style linear-beta variance-
            preserving schedule.
        score_parameterization : str
            Score-model regression target and score reconstruction strategy. Currently supported:
            "noise", "x0", and "edm".
        noise_features : str
            Noise/time feature representation passed to LightGBM. Currently supported:
            "raw_time" and "raw_time_log_std".
        edm_sigma_data : float
            Data standard deviation used by the EDM preconditioning coefficients when
            `score_parameterization="edm"`. The default 1.0 matches Treeffuser's
            standardized target scale.
        loss_weighting : {"uniform", "min_snr"}
            Per-sample weighting of the score-model regression loss. "uniform" (default)
            preserves the current Treeffuser behavior. "min_snr" applies the parameterization-
            aware min-SNR-gamma weighting from Hang et al. (2023) to cap the contribution
            of small-noise rows where the score is unboundedly large.
        min_snr_gamma : float
            Cap on the signal-to-noise ratio used when `loss_weighting="min_snr"`. Ignored
            otherwise. Standard practice is a value in the range 1-5.
        t_sampling : {"uniform", "log_sigma_normal", "log_snr_normal"}
            Distribution used to draw the training-time `t` values. "uniform" (default)
            keeps the historical Treeffuser behavior (with a small random anchor at t=1
            controlled by `uniform_endpoint_fraction` under flow matching).
            "log_sigma_normal" follows Karras et al. (EDM) by drawing
            `log sigma(t) ~ Normal(log_sigma_p_mean, log_sigma_p_std)` and inverting
            back to `t`; under flow matching this samples log of the path's noise
            coefficient beta(t) (so values are clipped at log beta = 0 since beta <= 1).
            "log_snr_normal" (flow-matching only) draws `log(alpha(t)/beta(t)) ~ Normal`
            instead; log-SNR has full real-line range so the Normal is not clipped.
            For tree-based learners these density-shift levers are stronger than loss
            weighting because they directly move histogram-bin density.
        uniform_endpoint_fraction : float
            Fraction of uniform-sampled training rows whose `t` is randomly set to
            exactly 1 (endpoint anchor). Only active under flow matching with
            `t_sampling="uniform"`. Default 0.05.
        log_sigma_p_mean, log_sigma_p_std : float
            Mean and standard deviation of the log-noise distribution used by
            `t_sampling="log_sigma_normal"` (score: log sigma; flow matching: log beta).
            Also reused by `"log_snr_normal"` for the log-SNR axis. EDM defaults (-1.2,
            1.2) are intended for the log-sigma axis; for log_snr_normal a more typical
            choice is `(p_mean=0, p_std=2)` centered around log SNR = 0.
        residualize : {"off", "mean", "mean_scale"}
            Optional conditional residualization before score-model fitting. "mean"
            subtracts a cross-fitted conditional mean, and "mean_scale" additionally
            divides by a cross-fitted conditional scale.
        residualize_k_folds : int
            Maximum number of folds used for cross-fitted residualization.
        seed : int
            Random seed for generating the training data and fitting the model.
        verbose : int
            Verbosity of the score model.
        """
        super().__init__(
            sde_initialize_from_data=sde_initialize_from_data,
            residualize=residualize,
            residualize_k_folds=residualize_k_folds,
            extra_residualizer_params=extra_residualizer_params,
        )
        self.sde_name = sde_name
        self.n_repeats = n_repeats
        self.n_estimators = n_estimators
        self.eval_percent = eval_percent
        self.early_stopping_rounds = early_stopping_rounds
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self.subsample_for_bin = subsample_for_bin
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.sde_initialize_from_data = sde_initialize_from_data
        self.sde_hyperparam_min = sde_hyperparam_min
        self.sde_hyperparam_max = sde_hyperparam_max
        self.training_objective = training_objective
        self.flow_path = flow_path
        self.score_parameterization = score_parameterization
        self.noise_features = noise_features
        self.edm_sigma_data = edm_sigma_data
        self.loss_weighting = loss_weighting
        self.min_snr_gamma = min_snr_gamma
        self.t_sampling = t_sampling
        self.uniform_endpoint_fraction = uniform_endpoint_fraction
        self.log_sigma_p_mean = log_sigma_p_mean
        self.log_sigma_p_std = log_sigma_p_std
        self.residualize = residualize
        self.residualize_k_folds = residualize_k_folds
        self.extra_lightgbm_params = extra_lightgbm_params or {}
        self.extra_residualizer_params = extra_residualizer_params or {}
        self._warn_on_flow_matching_ignored_params()

    def get_new_sde(self) -> DiffusionSDE:
        sde_cls = get_diffusion_sde(self.sde_name)
        assert not isinstance(sde_cls, dict)
        sde_kwargs = {}
        if self.sde_hyperparam_min is not None:
            sde_kwargs["hyperparam_min"] = self.sde_hyperparam_min
        if self.sde_hyperparam_max is not None:
            sde_kwargs["hyperparam_max"] = self.sde_hyperparam_max
        sde = sde_cls(**sde_kwargs)
        return sde

    def _warn_on_flow_matching_ignored_params(self) -> None:
        if self.training_objective != "flow_matching":
            return
        ignored_params = []
        if self.sde_name != "vesde":
            ignored_params.append("sde_name")
        if self.sde_initialize_from_data is not False:
            ignored_params.append("sde_initialize_from_data")
        if self.sde_hyperparam_min is not None:
            ignored_params.append("sde_hyperparam_min")
        if self.sde_hyperparam_max is not None:
            ignored_params.append("sde_hyperparam_max")
        if self.score_parameterization != "noise":
            ignored_params.append("score_parameterization")
        if self.edm_sigma_data != 1.0:
            ignored_params.append("edm_sigma_data")
        if self.loss_weighting != "uniform":
            ignored_params.append("loss_weighting")
        if self.min_snr_gamma != 5.0:
            ignored_params.append("min_snr_gamma")
        # t_sampling, log_sigma_p_mean, log_sigma_p_std are now used by the
        # flow-matching path too (log-beta-normal sampler reuses the same names).
        if not ignored_params:
            return
        warnings.warn(
            "training_objective='flow_matching' ignores score/SDE-only parameters: " f"{', '.join(ignored_params)}.",
            UserWarning,
            stacklevel=2,
        )

    def get_new_score_model(self) -> ScoreModel:
        score_model = LightGBMScoreModel(
            n_repeats=self.n_repeats,
            n_estimators=self.n_estimators,
            eval_percent=self.eval_percent,
            early_stopping_rounds=self.early_stopping_rounds,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_bin=self.max_bin,
            subsample_for_bin=self.subsample_for_bin,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            verbose=self.verbose,
            seed=self.seed,
            n_jobs=self.n_jobs,
            score_parameterization=self.score_parameterization,
            noise_features=self.noise_features,
            edm_sigma_data=self.edm_sigma_data,
            loss_weighting=self.loss_weighting,
            min_snr_gamma=self.min_snr_gamma,
            t_sampling=self.t_sampling,
            log_sigma_p_mean=self.log_sigma_p_mean,
            log_sigma_p_std=self.log_sigma_p_std,
            **self.extra_lightgbm_params,
        )
        return score_model

    def get_new_velocity_model(self) -> VelocityModel:
        velocity_model = LightGBMVelocityModel(
            n_repeats=self.n_repeats,
            n_estimators=self.n_estimators,
            eval_percent=self.eval_percent,
            early_stopping_rounds=self.early_stopping_rounds,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_bin=self.max_bin,
            subsample_for_bin=self.subsample_for_bin,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            verbose=self.verbose,
            seed=self.seed,
            n_jobs=self.n_jobs,
            flow_path=self.flow_path,
            noise_features=self.noise_features,
            t_sampling=self.t_sampling,
            log_sigma_p_mean=self.log_sigma_p_mean,
            log_sigma_p_std=self.log_sigma_p_std,
            uniform_endpoint_fraction=self.uniform_endpoint_fraction,
            **self.extra_lightgbm_params,
        )
        return velocity_model

    @property
    def n_estimators_true(self) -> list[int]:
        """
        The number of estimators that are actually used in the models (after early stopping),
        one for each dimension of the learned score or velocity (i.e. the dimension of y).
        """
        if self.training_objective == "flow_matching":
            assert isinstance(self.velocity_model, LightGBMVelocityModel)
            assert self.velocity_model.n_estimators_true is not None
            return self.velocity_model.n_estimators_true
        assert isinstance(self.score_model, LightGBMScoreModel)
        assert self.score_model.n_estimators_true is not None
        return self.score_model.n_estimators_true
