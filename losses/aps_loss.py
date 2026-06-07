import jax.numpy as jnp
import chex

Array = chex.Array
Numeric = chex.Numeric


def aps_loss(
    s_t: Array,
    task: Array,
) -> Array:
    """
    Following the APS loss defined in the paper "APS: Active Pretraining with Successor Features paper."
    https://arxiv.org/abs/2108.13956
    Args:
        s_t: state representation at time t.
        task: current task

    Returns:
        negative dot product of s_t and task

    """
    chex.assert_rank(
        [s_t, task],
        [1, 1],
    )
    chex.assert_type(
        [
            s_t,
            task,
        ],
        [float, float],
    )

    loss = -jnp.dot(s_t, task)

    return loss
