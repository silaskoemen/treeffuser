import abc
from collections.abc import Callable

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from treeffuser.sde.base_sde import BaseSDE


class FlowPath(abc.ABC):
    """
    Probability path used for flow matching.

    The first implementation uses the fixed time interval [0, 1]. Keep the path
    object small until experiments justify schedule variants.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def sample_prior(
        self,
        shape: tuple[int, ...],
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> Float[ndarray, "*shape"]:
        pass

    @abc.abstractmethod
    def interpolate(
        self,
        y0: Float[ndarray, "batch y_dim"],
        z: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        pass

    @abc.abstractmethod
    def target_velocity(
        self,
        y0: Float[ndarray, "batch y_dim"],
        z: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        pass

    @abc.abstractmethod
    def implied_score(
        self,
        y_t: Float[ndarray, "batch y_dim"],
        velocity: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        """Score of the marginal `p_t(y_t)` implied by the learned velocity."""

    @abc.abstractmethod
    def noise_scale(
        self,
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch 1"]:
        """beta(t), the conditional std of y_t given y_0 under z ~ N(0, I).
        Used by log-noise t-sampling: log beta(t) is the FM analog of the
        log-sigma sampling that drives histogram-bin density on the score side."""

    @abc.abstractmethod
    def signal_scale(
        self,
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch 1"]:
        """alpha(t), the coefficient of y_0 in y_t = alpha(t) y_0 + beta(t) z.
        Used alongside `noise_scale` to compute log-SNR = log(alpha/beta), the
        unbounded-range t-sampling key analogous to score-side log sigma."""


class LinearFlowPath(FlowPath):
    """
    Linear rectified-flow path from data to a standard-normal prior:

        y_t = (1 - t) y0 + t z
        dy_t / dt = z - y0
    """

    @property
    def name(self) -> str:
        return "linear"

    def sample_prior(
        self,
        shape: tuple[int, ...],
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> Float[ndarray, "*shape"]:
        if rng is None:
            rng = np.random.default_rng(seed)
        return rng.normal(size=shape)

    def interpolate(
        self,
        y0: Float[ndarray, "batch y_dim"],
        z: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        return (1.0 - t) * y0 + t * z

    def target_velocity(
        self,
        y0: Float[ndarray, "batch y_dim"],
        z: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        return z - y0

    def implied_score(
        self,
        y_t: Float[ndarray, "batch y_dim"],
        velocity: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        # For linear FM with z ~ N(0, I), Tweedie + the identity
        # E[y0 | y_t] = y_t - t v gives score = -(y_t + (1 - t) v) / t.
        # The 1/t factor diverges at t=0 for approximate v, so any stochastic
        # sampler that uses this score must vanish stochasticity at t=0.
        return -(y_t + (1.0 - t) * velocity) / t

    def noise_scale(self, t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return t

    def signal_scale(self, t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return 1.0 - t


class TrigFlowPath(FlowPath):
    """
    Variance-preserving cosine-sine path between data and a standard-normal prior:

        y_t = cos(pi t / 2) y0 + sin(pi t / 2) z
        dy_t / dt = (pi/2) (-sin(pi t / 2) y0 + cos(pi t / 2) z)

    Satisfies a(t)^2 + b(t)^2 = 1 for all t (variance-preserving) and reaches
    exact prior at t=1 (a(1)=0, b(1)=1). Wronskian W(t) = pi/2 is constant.
    """

    @property
    def name(self) -> str:
        return "trig"

    def sample_prior(
        self,
        shape: tuple[int, ...],
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> Float[ndarray, "*shape"]:
        if rng is None:
            rng = np.random.default_rng(seed)
        return rng.normal(size=shape)

    def _coeffs(self, t: Float[ndarray, "batch 1"]):
        phi = np.pi * t / 2.0
        a = np.cos(phi)
        b = np.sin(phi)
        a_prime = -(np.pi / 2.0) * b
        b_prime = (np.pi / 2.0) * a
        return a, b, a_prime, b_prime

    def interpolate(
        self,
        y0: Float[ndarray, "batch y_dim"],
        z: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        a, b, _, _ = self._coeffs(t)
        return a * y0 + b * z

    def target_velocity(
        self,
        y0: Float[ndarray, "batch y_dim"],
        z: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        _, _, a_prime, b_prime = self._coeffs(t)
        return a_prime * y0 + b_prime * z

    def implied_score(
        self,
        y_t: Float[ndarray, "batch y_dim"],
        velocity: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        # General formula: score = (a' y_t - a v) / (W b), W = pi/2 for trig.
        # Simplifies to score = -y_t - (2/pi) * cot(pi t / 2) * v.
        # Same 1 / sin(pi t / 2) singularity at t=0 as linear; vanishing eps(t)
        # at t=0 makes the (eps^2 / 2) * score term well-behaved.
        a, b, a_prime, _ = self._coeffs(t)
        return (a_prime * y_t - a * velocity) / ((np.pi / 2.0) * b)

    def noise_scale(self, t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return np.sin(np.pi * t / 2.0)

    def signal_scale(self, t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return np.cos(np.pi * t / 2.0)


class VPFlowPath(FlowPath):
    """
    Variance-preserving path with the linear-beta DDPM schedule:

        T(t)         = (1/2) beta_min t + (1/4) (beta_max - beta_min) t^2
        alpha_bar(t) = exp(-T(t))
        a(t)         = sqrt(alpha_bar(t))
        b(t)         = sqrt(1 - alpha_bar(t))

    At t=0: a=1, b=0. At t=1: a is small but nonzero (e.g. ~0.08 with
    beta_max=20). For standardized residualized data with E[y0]=0 and
    Var[y0]=1 the t=1 marginal matches N(0, I) in its first two moments
    (mean=0 and variance = a^2 * Var[y0] + b^2 = 1), so initializing sampling
    from N(0, I) introduces no first- or second-order bias; higher-order
    moments coincide only when y0 itself is Gaussian, but the practical error
    from this approximation is below the sampler's discretization error
    across all benchmarks we ran.

    Near t=0, b(t) ~ sqrt(t) and b'(t) ~ 1 / sqrt(t), so velocity targets have
    higher variance at small t than linear/trig. The (eps(t)^2 / 2) * score
    correction is still bounded for eps(t) vanishing as t^k (k >= 1).
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        if beta_min <= 0 or beta_max <= beta_min:
            raise ValueError("Require 0 < beta_min < beta_max.")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    @property
    def name(self) -> str:
        return "vp"

    def sample_prior(
        self,
        shape: tuple[int, ...],
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> Float[ndarray, "*shape"]:
        if rng is None:
            rng = np.random.default_rng(seed)
        return rng.normal(size=shape)

    def _T(self, t: Float[ndarray, "batch 1"]):
        return 0.5 * self.beta_min * t + 0.25 * (self.beta_max - self.beta_min) * t**2

    def _T_prime(self, t: Float[ndarray, "batch 1"]):
        return 0.5 * (self.beta_min + (self.beta_max - self.beta_min) * t)

    def _coeffs(self, t: Float[ndarray, "batch 1"]):
        alpha_bar = np.exp(-self._T(t))
        a = np.sqrt(alpha_bar)
        b_sq = np.clip(1.0 - alpha_bar, 0.0, 1.0)
        b = np.sqrt(b_sq)
        T_prime = self._T_prime(t)
        a_prime = -0.5 * T_prime * a
        # 2 b b' = -alpha_bar' = T' alpha_bar  =>  b' = T' alpha_bar / (2 b)
        # Use small floor to avoid division by zero at exact t=0; training/sampling
        # use t >= eps so this branch is exercised only defensively.
        b_safe = np.where(b > 1e-12, b, 1e-12)
        b_prime = T_prime * alpha_bar / (2.0 * b_safe)
        return a, b, a_prime, b_prime

    def interpolate(
        self,
        y0: Float[ndarray, "batch y_dim"],
        z: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        a, b, _, _ = self._coeffs(t)
        return a * y0 + b * z

    def target_velocity(
        self,
        y0: Float[ndarray, "batch y_dim"],
        z: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        _, _, a_prime, b_prime = self._coeffs(t)
        return a_prime * y0 + b_prime * z

    def implied_score(
        self,
        y_t: Float[ndarray, "batch y_dim"],
        velocity: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> Float[ndarray, "batch y_dim"]:
        a, b, a_prime, b_prime = self._coeffs(t)
        W = a * b_prime - a_prime * b
        return (a_prime * y_t - a * velocity) / (W * b)

    def noise_scale(self, t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        _, b, _, _ = self._coeffs(t)
        return b

    def signal_scale(self, t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        a, _, _, _ = self._coeffs(t)
        return a


def get_flow_path(path: str | FlowPath) -> FlowPath:
    if isinstance(path, FlowPath):
        return path
    if path == "linear":
        return LinearFlowPath()
    if path == "trig":
        return TrigFlowPath()
    if path == "vp":
        return VPFlowPath()
    raise ValueError(f"Unknown flow path: {path!r}")


class ReverseVelocityODE(BaseSDE):
    """
    Reverse-time deterministic ODE induced by a learned flow-matching velocity.

    If the forward path follows dy/dt = v(y, t), sampling uses reverse time
    s = 1 - t, so dy/ds = -v(y, 1 - s).
    """

    def __init__(
        self,
        velocity_fn: Callable[
            [Float[ndarray, "batch y_dim"], Float[ndarray, "batch 1"]],
            Float[ndarray, "batch y_dim"],
        ],
        t_reverse_origin: float = 1.0,
    ):
        self.velocity_fn = velocity_fn
        self.t_reverse_origin = t_reverse_origin

    def drift_and_diffusion(
        self,
        y: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        forward_t = self.t_reverse_origin - t
        drift = -self.velocity_fn(y, forward_t)
        diffusion = np.zeros_like(y)
        return drift, diffusion

    def __repr__(self):
        return f"ReverseVelocityODE(t_origin={self.t_reverse_origin}, velocity_fn={self.velocity_fn})"


StochasticitySchedule = Callable[[Float[ndarray, "batch 1"]], Float[ndarray, "batch 1"]]


def linear_stochasticity_schedule(stochasticity: float) -> StochasticitySchedule:
    """eps(t) = stochasticity * t. Vanishes at t=0 (matches sigma(t) for linear FM).
    Monotone increasing — injects more noise toward the prior end."""

    def schedule(forward_t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return stochasticity * forward_t

    return schedule


def quadratic_stochasticity_schedule(stochasticity: float) -> StochasticitySchedule:
    """eps(t) = stochasticity * t^2. Vanishes at t=0 and concentrates noise more
    sharply at high t than the linear schedule — useful when tail-coverage misses
    are tail-concentrated (the score's 1/t factor is larger at small t, so a
    quadratic schedule keeps the (eps^2 / 2) * score term bounded near data)."""

    def schedule(forward_t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return stochasticity * forward_t**2

    return schedule


def sqrt_stochasticity_schedule(stochasticity: float) -> StochasticitySchedule:
    """eps(t) = stochasticity * sqrt(t). Vanishes at t=0 but less sharply than linear,
    so the schedule injects more noise near the data endpoint. Useful as a control
    that puts more mass in the low-t regime."""

    def schedule(forward_t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return stochasticity * np.sqrt(forward_t)

    return schedule


def tent_stochasticity_schedule(stochasticity: float) -> StochasticitySchedule:
    """eps(t) = stochasticity * t * (1 - t). Vanishes at both endpoints, peaks at
    t=0.5. Concentrates stochasticity in mid-path where samples cross between modes;
    intended as a control for the hypothesis that tail-coverage gain requires high-t
    noise rather than mid-path noise."""

    def schedule(forward_t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return stochasticity * forward_t * (1.0 - forward_t)

    return schedule


_STOCHASTICITY_SCHEDULE_BUILDERS = {
    "linear": linear_stochasticity_schedule,
    "quadratic": quadratic_stochasticity_schedule,
    "sqrt": sqrt_stochasticity_schedule,
    "tent": tent_stochasticity_schedule,
}


def get_stochasticity_schedule(shape: str, stochasticity: float) -> StochasticitySchedule:
    if shape not in _STOCHASTICITY_SCHEDULE_BUILDERS:
        raise ValueError(
            f"Unknown stochasticity schedule shape: {shape!r}. "
            f"Available: {sorted(_STOCHASTICITY_SCHEDULE_BUILDERS.keys())}."
        )
    return _STOCHASTICITY_SCHEDULE_BUILDERS[shape](stochasticity)


class ReverseVelocityInterpolant(BaseSDE):
    """
    Reverse-time stochastic interpolant SDE induced by a learned velocity.

    Given a velocity field `v(y, t)` and the implied marginal score `s(y, t)` from
    a Gaussian flow path, the family of marginal-preserving SDEs parameterized by
    a non-negative stochasticity schedule `ε(t)` is

        dy = (v + (ε² / 2) s) dt + ε dW       (forward, data -> prior)

    Reversing time gives the sampling SDE on `s_rev = T - t`:

        dy = (-v + (ε² / 2) s) ds_rev + ε dW̃   (reverse, prior -> data)

    Setting `stochasticity=0` reduces this to `ReverseVelocityODE` exactly.

    See Albergo, Boffi & Vanden-Eijnden (2023), "Stochastic Interpolants:
    A Unifying Framework for Flows and Diffusions."
    """

    def __init__(
        self,
        velocity_fn: Callable[
            [Float[ndarray, "batch y_dim"], Float[ndarray, "batch 1"]],
            Float[ndarray, "batch y_dim"],
        ],
        flow_path: FlowPath,
        stochasticity_schedule: StochasticitySchedule,
        t_reverse_origin: float = 1.0,
    ):
        self.velocity_fn = velocity_fn
        self.flow_path = flow_path
        self.stochasticity_schedule = stochasticity_schedule
        self.t_reverse_origin = t_reverse_origin

    def drift_and_diffusion(
        self,
        y: Float[ndarray, "batch y_dim"],
        t: Float[ndarray, "batch 1"],
    ) -> tuple[Float[ndarray, "batch y_dim"], Float[ndarray, "batch y_dim"]]:
        forward_t = self.t_reverse_origin - t
        velocity = self.velocity_fn(y, forward_t)
        eps = self.stochasticity_schedule(forward_t)
        # Broadcast to a per-row column when the schedule returns a scalar/(batch, 1).
        eps = np.broadcast_to(np.asarray(eps), (*y.shape[:1], 1))
        if not np.any(eps):
            return -velocity, np.zeros_like(y)
        score = self.flow_path.implied_score(y_t=y, velocity=velocity, t=forward_t)
        drift = -velocity + 0.5 * (eps**2) * score
        diffusion = np.broadcast_to(eps, y.shape)
        return drift, diffusion

    def __repr__(self):
        return (
            f"ReverseVelocityInterpolant(t_origin={self.t_reverse_origin}, "
            f"flow_path={self.flow_path.name}, velocity_fn={self.velocity_fn})"
        )
