"""
Consolidation loss without temporal difference loss.
"""

import jax
import chex
import rlax
import jax.numpy as jnp

Array = chex.Array
Numeric = chex.Numeric


def q_values_consolidation_loss(
    q_tm1: Array,
    a_tm1: Numeric,
    g_flow: Array,
    capacity: Array,
    grad_error_bound: Numeric,
    allow_back_flow: Array,
) -> Numeric:
    """

    This function only compute the consolidation loss using the q values.
    u_0 is the first beaker and u_k is the last beaker.

    u_0 <---> g_flow[0]--- u_1
    u_k-1 ---g_flow[k-1]<--> u_k <---> g_flow[k]--- u_k+1
    u_k-1 ---g_flow[k-1]<--> u_k <---> g_flow[k]--- (u_k+1 = 0)

    Args:
        q_tm1: Q-value of beakers [u1, u2, u3, ...,] at time t-1 (shape: [num_beakers, num_actions])
        a_tm1: action index at time t-1.
        g_flow: flow variables between beakers (shape: [num_beakers, 1])
        capacity: capacity of beakers (shape: [num_beakers, 1])
        grad_error_bound: gradient error bound
        update_step: current update step

    Returns:
        loss for q_learning with consolidation using the beakers setup

    """
    chex.assert_rank(
        [q_tm1, a_tm1, g_flow, capacity, grad_error_bound, allow_back_flow],
        [2, 0, 1, 1, 0, 1],
    )
    chex.assert_type(
        [q_tm1, a_tm1, g_flow, capacity, grad_error_bound, allow_back_flow],
        [float, int, float, int, float, bool],
    )

    num_beakers: int = capacity.shape[0]

    loss = 0

    for i in range(num_beakers):
        # need special care for the first beaker as it has no incoming flow
        if i == 0:
            q_tm1_u1 = q_tm1[i]
            q_tm1_u2 = q_tm1[i + 1]
            loss += (
                g_flow[i]
                * rlax.l2_loss(
                    rlax.clip_gradient(
                        jax.lax.stop_gradient(q_tm1_u2[a_tm1]) - q_tm1_u1[a_tm1],
                        -grad_error_bound,
                        grad_error_bound,
                    )  # flow from beaker 2 to beaker 1
                )
            ) / capacity[i]

        # need special care for the last beaker as it has no outgoing flow
        elif i == num_beakers - 1:
            q_tm1_last_u = q_tm1[i]
            q_tm1_second_last_u = q_tm1[i - 1]
            loss += (
                g_flow[i]
                * rlax.l2_loss(
                    rlax.clip_gradient(
                        -q_tm1_last_u[a_tm1], -grad_error_bound, grad_error_bound
                    )  # flow from the last beaker to leak
                )  # flow from the last beaker to leak
                + g_flow[i - 1]
                * rlax.l2_loss(
                    rlax.clip_gradient(
                        q_tm1_last_u[a_tm1]
                        - jax.lax.stop_gradient(
                            q_tm1_second_last_u[
                                a_tm1
                            ],  # flow from the second last beaker to the last beaker
                        ),
                        -grad_error_bound,
                        grad_error_bound,
                    )
                )
            ) / capacity[i]

        # for all other beakers we have both incoming and outgoing flow
        else:
            q_tm1_current = q_tm1[i]
            q_tm1_next = q_tm1[i + 1]
            q_tm1_previous = q_tm1[i - 1]

            """
            Add a time step check to determine if back flow should be considered. 
            If the time scale if less than 2**i / g_flow[0], then back flow will not be added.
            """

            if allow_back_flow[i]:

                loss += (
                    g_flow[i]
                    * rlax.l2_loss(
                        rlax.clip_gradient(
                            jax.lax.stop_gradient(q_tm1_next[a_tm1])
                            - q_tm1_current[a_tm1],
                            -grad_error_bound,
                            grad_error_bound,
                        )  # flow from beaker k+1 to beaker k
                    )
                    / capacity[i]
                )

            loss += (
                g_flow[i - 1]
                * rlax.l2_loss(
                    rlax.clip_gradient(
                        q_tm1_current[a_tm1]
                        - jax.lax.stop_gradient(
                            q_tm1_previous[a_tm1]  # flow from beaker k to beaker k-1
                        ),
                        -grad_error_bound,
                        grad_error_bound,
                    )
                )
            ) / capacity[i]

    return loss


def q_values_consolidation_loss_without_back_flow(
    q_tm1: Array,
    a_tm1: Numeric,
    g_flow: Array,
    capacity: Array,
    grad_error_bound: Numeric,
) -> Numeric:
    """

    This function only compute the consolidation loss using the q values with back flow from the beaker after it.
    This function is only used for the first task.

    u_k-1 ---> g_flow[k-1]---> u_k


    Args:
        q_tm1: Q-value of beakers [u1, u2, u3, ...,] at time t-1 (shape: [num_beakers, num_actions])
        a_tm1: action index at time t-1.
        g_flow: flow variables between beakers (shape: [num_beakers, 1])
        capacity: capacity of beakers (shape: [num_beakers, 1])
        grad_error_bound: gradient error bound

    Returns:
        loss for q_learning with consolidation using the beakers setup

    """
    chex.assert_rank(
        [q_tm1, a_tm1, g_flow, capacity, grad_error_bound],
        [2, 0, 1, 1, 0],
    )
    chex.assert_type(
        [q_tm1, a_tm1, g_flow, capacity, grad_error_bound],
        [float, int, float, int, float],
    )

    num_beakers = capacity.shape[0]

    loss = 0

    for i in range(num_beakers):
        # ignore the first beaker as it has no incoming flow during the first task
        if i == 0:
            continue

        # for all other beakers we only consider the beaker before it during the first task
        else:
            q_tm1_current = q_tm1[i]
            q_tm1_previous = q_tm1[i - 1]
            loss += (
                g_flow[i - 1]
                * rlax.l2_loss(
                    rlax.clip_gradient(
                        q_tm1_current[a_tm1]
                        - jax.lax.stop_gradient(
                            q_tm1_previous[a_tm1]  # flow from beaker k to beaker k-1
                        ),
                        -grad_error_bound,
                        grad_error_bound,
                    )
                )
            ) / capacity[i]

    return loss
