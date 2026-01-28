import jax.numpy as jnp
import optax


def warmup_stable_decay_schedule(
    peak_value: float,
    total_steps: int,
    warmup_frac: float,
    decay_frac: float,
    end_value: float = 0.0,
) -> optax.Schedule:
    """Creates a Warmup-Stable-Decay learning rate schedule with sqrt decay.

    Args:
        peak_value: Peak learning rate during stable phase.
        total_steps: Total number of training steps.
        warmup_frac: Fraction of total steps for linear warmup.
        decay_frac: Fraction of total steps for sqrt decay phase (cooldown).
        end_value: Final learning rate value (default 0.0).

    Returns:
        Schedule function mapping step counts to learning rates.
    """
    warmup_steps = int(total_steps * warmup_frac)
    decay_steps = int(total_steps * decay_frac)
    stable_steps = total_steps - warmup_steps - decay_steps

    def sqrt_decay(count: int) -> float:
        progress = jnp.minimum(count / decay_steps, 1.0)
        return peak_value * (1 - jnp.sqrt(progress)) + end_value * jnp.sqrt(progress)

    schedules = [
        optax.linear_schedule(init_value=0.0, end_value=peak_value, transition_steps=warmup_steps),
        optax.constant_schedule(peak_value),
        sqrt_decay,
    ]

    boundaries = [warmup_steps, warmup_steps + stable_steps]

    return optax.join_schedules(schedules, boundaries)
