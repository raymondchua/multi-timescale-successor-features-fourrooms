from typing import Any, Callable, Mapping, Text, Tuple, Sequence, List

import chex
import jax
import jax.numpy as jnp

from functools import partial
from flax.core import FrozenDict
from helpers.replay_lib import Transition_Task
import flax

Array = chex.Array
Params = FrozenDict
Network = flax.linen.Module

@partial(jax.jit, static_argnames=("num_beakers", "network"))
def get_all_sf(
    transitions: Transition_Task,
    num_beakers: int,
    network_params: Sequence[Params],
    network: Network,
) -> Array:
    """
    Computes the successor features (SFs) for all beakers.

    Parameters
    ----------
    transitions: Transition_Task
        The transitions containing s_t and task.
    num_beakers: int
        Number of beakers.
    network_params: Sequence[Params]
        Parameters for each network.
    network: Network
        The network to apply.

    Returns
    -------
    Array
        Successor features with shape (batch_size, num_beakers, sf_dim, num_actions).
    """
    # Stack all parameters for vmapping
    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *network_params[:num_beakers]
    )

    # Define a batched apply function
    def single_apply(params, s_t, task):
        return network.apply({"params": params}, s_t, task).sf

    # vmap over parameters; s_t and task are broadcasted
    batched_apply = jax.vmap(single_apply, in_axes=(0, None, None))
    sf_all = batched_apply(stacked_params, transitions.s_t, transitions.task)

    # Swap axes to match the desired shape
    sf_all = jnp.swapaxes(
        sf_all, 0, 1
    )  # shape: (batch_size, num_beakers, sf_dim, num_actions)

    return sf_all
