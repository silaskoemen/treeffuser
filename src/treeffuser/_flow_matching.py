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


def get_flow_path(path: str | FlowPath) -> FlowPath:
    if isinstance(path, FlowPath):
        return path
    if path == "linear":
        return LinearFlowPath()
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
    """Default eps(t) = stochasticity * t. Vanishes at t=0 (matches sigma(t) for linear FM)."""

    def schedule(forward_t: Float[ndarray, "batch 1"]) -> Float[ndarray, "batch 1"]:
        return stochasticity * forward_t

    return schedule


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
