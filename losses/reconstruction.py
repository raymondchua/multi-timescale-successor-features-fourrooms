import jax.numpy as jnp
import chex

Array = chex.Array
Numeric = chex.Numeric


def next_state_prediction_error(
    s_t: Array,
    predict_s_t: Array,
) -> Array:
    """

    Args:
        s_t: observed next state after performing an action
        predict_s_t: predicted next state from the reconstruction network

    Returns:
        the l2 loss between predicted next state and the observed next state

    """
    chex.assert_rank(
        [s_t, predict_s_t],
        [3, 3],
    )
    chex.assert_type(
        [s_t, predict_s_t],
        [int, float],
    )

    predict_s_t_flatten = jnp.reshape(predict_s_t, (-1))
    s_t = s_t.astype(jnp.float32) / 255.0  # normalize the input
    s_t_flatten = jnp.reshape(s_t, (-1))
    error = predict_s_t_flatten - s_t_flatten

    return error
