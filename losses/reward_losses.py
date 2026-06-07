import chex
import rlax
import jax.numpy as jnp

Array = chex.Array
Numeric = chex.Numeric


def reward_prediction_loss(
    s_t: Array,
    task: Array,
    r_t: Numeric,
) -> Array:
    """

    Args:
        s_t: state representation at time t.
        task: current task
        r_t: reward at time t

    Returns:
        the l2 loss between predicted reward and the ground truth reward value from the environment

    """
    chex.assert_rank(
        [s_t, task, r_t],
        [1, 1, 0],
    )
    chex.assert_type(
        [s_t, task, r_t],
        [float, float, float],
    )
    loss = rlax.l2_loss(jnp.dot(s_t, task) - r_t)

    return loss
