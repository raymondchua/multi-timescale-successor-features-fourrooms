import jax
import jax.numpy as jnp
import chex
import rlax
from absl import logging
import jax.tree_util as tree_util
from jax import lax
from typing import Any, Tuple
from flax.core import FrozenDict

Array = chex.Array
Numeric = chex.Numeric
Params = FrozenDict


def sf_consolidation_loss(
    sf_1_tm1: Array,
    a_tm1: Numeric,
    g_flow: Array,
    capacity: Array,
    mask_condition: Array,
) -> Numeric:
    """

    u_0 <---> g_flow[0]--- u_1
    u_k-1 ---g_flow[k-1]<--> u_k <---> g_flow[k]--- u_k+1
    u_k-1 ---g_flow[k-1]<--> u_k <---> g_flow[k]--- (u_k+1 = 0)

    Args:
        sf_1_tm1: successor features of beakers [u1, u2, u3, ...,] at time t-1
                    (shape: [num_beakers, num_actions, sf_dim])
        a_tm1: action at time t-1 (shape: [num_actions])
        g_flow: flow variables between beakers (shape: [num_beakers, num_actions, sf_dim])
        capacity: capacity of beakers (shape: [num_beakers, num_actions, sf_dim])
        mask_condition: mask of beakers (shape: [num_beakers - 1, num_actions, sf_dim])

    Returns:
        SF consolidation loss without using sf-td error. This is for the case which the
        SFs are updated using the classic Q-learning td error.

    """
    chex.assert_rank(
        [
            sf_1_tm1,
            a_tm1,
            g_flow,
            capacity,
            mask_condition,
        ],
        [3, 0, 3, 3, 3],
    )
    chex.assert_type(
        [
            sf_1_tm1,
            a_tm1,
            g_flow,
            capacity,
            mask_condition,
        ],
        [float, int, float, int, int],
    )

    """
    For the first beaker, there is no incoming flow.
    """
    g_first = g_flow[0, :, :]
    capacity_first = capacity[0, :, :]

    var_tm1_u1 = sf_1_tm1[0, :, :]
    var_tm1_u2 = sf_1_tm1[1, :, :]

    loss_first = rlax.l2_loss(var_tm1_u1 - jax.lax.stop_gradient(var_tm1_u2))
    loss_first *= g_first / capacity_first

    """
    For the last beaker, the outgoing flow is to the leak.
    """

    g_last = g_flow[-1, :, :]
    g_second_last = g_flow[-2, :, :]
    capacity_last = capacity[-1, :, :]

    var_tm1_last = sf_1_tm1[-1, :, :]
    var_tm1_second_last = sf_1_tm1[-2, :, :]

    # u_k-1 ---g_flow[k-1]<--> u_k <---> g_flow[k]--- (u_k+1 = 0)
    loss_last = rlax.l2_loss(jnp.zeros_like(var_tm1_last) - var_tm1_last)

    loss_last *= g_last / capacity_last

    loss_second_last = rlax.l2_loss(
        var_tm1_last - jax.lax.stop_gradient(var_tm1_second_last)
    )

    loss_second_last *= g_second_last / capacity_last

    """
    For the rest of the beakers, there is both incoming and outgoing flow.
    """

    g_flow_second_to_second_last = g_flow[:-1, :, :]
    capacity_second_to_second_last_forward = capacity[1:, :, :]
    capacity_second_to_second_last_backward = capacity[:-1, :, :]

    var_tm1_previous = sf_1_tm1[:-1, :]  # from 0 to num_beakers-2
    var_tm1_current = sf_1_tm1[1:, :]  # from 1 to num_beakers-1

    """
    u_k-1 ---g_flow[k-1]--> u_k. Capacity has to match the flow direction, so starting from k = 1 since the first 
    beaker has no incoming flow since index=- 1 does not exist.
    """
    loss_second_to_second_last_forward = rlax.l2_loss(
        jax.lax.stop_gradient(var_tm1_previous) - var_tm1_current,
    )
    loss_second_to_second_last_forward *= (
        g_flow_second_to_second_last / capacity_second_to_second_last_forward
    )

    """
    u_k <--- g_flow[k]--- u_k+1
    Make mask to ensure that the timescales are respected. Capacity has to match the flow direction, so starting 
    from k = 0 and K = num_beakers - 1 since the last beaker has no flow from the next beaker.
    """

    loss_second_to_second_last_backwards = rlax.l2_loss(
        jax.lax.stop_gradient(var_tm1_current) - var_tm1_previous
    )

    g_flow_second_to_second_last *= (
        mask_condition  # mask the back flow to ensure that the timescales are respected
    )

    loss_second_to_second_last_backwards *= (
        g_flow_second_to_second_last / capacity_second_to_second_last_backward
    )

    loss = (
        loss_first
        + loss_last
        + loss_second_last
        + loss_second_to_second_last_forward
        + loss_second_to_second_last_backwards
    )

    return jnp.mean(loss)


def sf_consolidation_loss_without_back_flow(
    sf_1_tm1: Array,
    a_tm1: Numeric,
    g_flow: Array,
    capacity: Array,
) -> Numeric:
    """

    u_k-1 ---> g_flow[k-1]---> u_k

    Args:
        sf_1_tm1: successor features of beakers [u1, u2, u3, ...,] at time t-1
                (shape: [num_beakers, num_actions, sf_dim])
        a_tm1: action at time t-1
        g_flow: flow variables between beakers (shape: [num_beakers, num_actions, sf_dim])
        capacity: capacity of beakers (shape: [num_beakers, num_actions, sf_dim])

    Returns:
        SF consolidation loss without using sf-td error. This is for the case which the
        SFs are updated using the classic Q-learning td error.

    """

    chex.assert_rank(
        [
            sf_1_tm1,
            a_tm1,
            g_flow,
            capacity,
        ],
        [3, 0, 3, 3],
    )
    chex.assert_type(
        [
            sf_1_tm1,
            a_tm1,
            g_flow,
            capacity,
        ],
        [
            float,
            int,
            float,
            int,
        ],
    )

    var_tm1_previous = sf_1_tm1[
        :-1, :, :
    ]  # ignore the last beaker as it has no outgoing flow
    var_tm1_current = sf_1_tm1[
        1:, :, :
    ]  # ignore the first beaker as it has no incoming flow

    g_flow = g_flow[:-1, :, :]  # ignore the last beaker as it has no outgoing flow
    capacity = capacity[1:, :, :]  # ignore the first beaker as it has no incoming flow

    """
    u_k-1 ---g_flow[k-1]--> u_k. Capacity has to match the flow direction, so starting from k = 1 since the first
    beaker has no incoming flow since index=- 1 does not exist.
    """

    # loss is the mean of the MSE loss between the current and previous beaker. The loss is weighted by the flow
    # and divided by the capacity of the beaker. The loss for the first beaker is ignored.

    loss = rlax.l2_loss(jax.lax.stop_gradient(var_tm1_previous) - var_tm1_current)
    loss *= g_flow / capacity

    return jnp.mean(loss)


def sf_consolidation_loss_without_back_flow_leak(
    sf_1_tm1: Array,
    a_tm1: Numeric,
    g_flow: Array,
    capacity: Array,
) -> Numeric:
    """

    u_k-1 ---> g_flow[k-1]---> u_k

    Main difference with sf_consolidation_loss_without_back_flow_leak is that the last beaker has a leak term,
    which promotes decay to induce forgetting.

    Args:
        sf_1_tm1: successor features of beakers [u1, u2, u3, ...,] at time t-1
                (shape: [num_beakers, num_actions, sf_dim])
        a_tm1: action at time t-1
        g_flow: flow variables between beakers (shape: [num_beakers, num_actions, sf_dim])
        capacity: capacity of beakers (shape: [num_beakers, num_actions, sf_dim])

    Returns:
        SF consolidation loss without using sf-td error. This is for the case which the
        SFs are updated using the classic Q-learning td error.

    """

    chex.assert_rank(
        [
            sf_1_tm1,
            a_tm1,
            g_flow,
            capacity,
        ],
        [3, 0, 3, 3],
    )
    chex.assert_type(
        [
            sf_1_tm1,
            a_tm1,
            g_flow,
            capacity,
        ],
        [
            float,
            int,
            float,
            int,
        ],
    )

    var_tm1_previous = sf_1_tm1[
        :-1, :, :
    ]  # ignore the last beaker as it has no outgoing flow
    var_tm1_current = sf_1_tm1[
        1:, :, :
    ]  # ignore the first beaker as it has no incoming flow

    g_flow = g_flow[:-1, :, :]  # ignore the last beaker as it has no outgoing flow
    capacity = capacity[1:, :, :]  # ignore the first beaker as it has no incoming flow

    """
    u_k-1 ---g_flow[k-1]--> u_k. Capacity has to match the flow direction, so starting from k = 1 since the first
    beaker has no incoming flow since index=- 1 does not exist.
    """

    # loss is the mean of the MSE loss between the current and previous beaker. The loss is weighted by the flow
    # and divided by the capacity of the beaker. The loss for the first beaker is ignored.

    loss = rlax.l2_loss(jax.lax.stop_gradient(var_tm1_previous) - var_tm1_current)
    loss *= g_flow / capacity

    """
    For the last beaker, the outgoing flow is to the leak.
    """

    g_last = g_flow[-1, :, :]
    capacity_last = capacity[-1, :, :]

    var_tm1_last = sf_1_tm1[-1, :, :]

    # u_k-1 ---g_flow[k-1]<--> u_k <---> g_flow[k]--- (u_k+1 = 0)
    loss_last = rlax.l2_loss(jnp.zeros_like(var_tm1_last) - var_tm1_last)

    loss_last *= g_last / capacity_last

    return jnp.mean(loss)


def consolidation_param_update(beaker0, beaker1, scale_factor):
    return tree_util.tree_map(lambda p0, p1: scale_factor * (p0 - p1), beaker0, beaker1)


def squared_sum(tree):
    return tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x), tree, initializer=0.0)

@jax.jit
def update_and_accumulate_tree(p1: Params, p2: Params, scale: float, loss: float) -> Tuple[Params, Array]:
    # Update parameters
    updated_tree = jax.tree_util.tree_map(lambda a, b: a + (scale * (b - a)), p1, p2)

    # Compute per-leaf squared L2 delta for loss accumulation
    loss_terms = jax.tree_util.tree_map(lambda a, b: jnp.sum(jnp.square(scale * (b - a))), p1, p2)
    total_loss = loss + jnp.sum(jnp.stack(jax.tree_util.tree_leaves(loss_terms)))
    return updated_tree, total_loss

@jax.jit
def pytree_l2_norm(p: Params) -> Array:
    """
    Computes the L2 norm of a PyTree.
    :param p:
    :return:
    """
    norm_terms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), p)
    return jnp.sum(jnp.stack(jax.tree_util.tree_leaves(norm_terms)))