from .transition_types import (
    Transition,
    Transition_Task,
    Transition_SF,
    Transition_Task_OneHotAction,
    Transition_Task_Prev_Action,
)

import jax.numpy as jnp


def build_n_step_transition(transitions):
    """Builds a single n-step transition from n 1-step transitions."""
    r_t = 0.0
    discount_t = 1.0
    for transition in transitions:
        r_t += discount_t * transition.r_t
        discount_t *= transition.discount_t

    # n-step transition, letting s_tm1 = s_tmn, and a_tm1 = a_tmn.
    return Transition(
        s_tm1=transitions[0].s_tm1,
        a_tm1=transitions[0].a_tm1,
        r_t=r_t,
        discount_t=discount_t,
        s_t=transitions[-1].s_t,
    )


def build_n_step_transition_task(transitions):
    """Builds a single n-step transition from n 1-step transitions."""
    r_t = 0.0
    discount_t = 1.0
    for transition in transitions:
        r_t += discount_t * transition.r_t
        discount_t *= transition.discount_t

    # n-step transition, letting s_tm1 = s_tmn, and a_tm1 = a_tmn.
    return Transition_Task(
        s_tm1=transitions[0].s_tm1,
        a_tm1=transitions[0].a_tm1,
        r_t=r_t,
        discount_t=discount_t,
        s_t=transitions[-1].s_t,
        task=transitions[-1].task,
    )

def build_n_step_transition_task_prev_action(transitions, a_tm2):
    """Builds a single n-step transition from n 1-step transitions including the addition of previous action.
    prev_action is the action taken at time step t-2.
    """
    r_t = 0.0
    discount_t = 1.0

    for transition in transitions:
        r_t += discount_t * transition.r_t
        discount_t *= transition.discount_t

    # n-step transition, letting s_tm1 = s_tmn, and a_tm1 = a_tmn.
    return Transition_Task_Prev_Action(
        s_tm1=transitions[0].s_tm1,
        a_tm1=transitions[0].a_tm1,
        a_tm2=a_tm2,
        r_t=r_t,
        discount_t=discount_t,
        s_t=transitions[-1].s_t,
        task=transitions[-1].task,
    )


def build_n_step_SF_transition(transitions, num_actions, a_t):
    """Builds a single n-step transition from n 1-step transitions."""
    r_t = 0.0
    discount_t = 1.0
    for transition in transitions:
        r_t += discount_t * transition.r_t
        discount_t *= transition.discount_t

    a_tm1_vector = jnp.zeros(num_actions)
    a_tm1_vector = a_tm1_vector.at[transitions[0].a_tm1].set(1)

    a_t_vector = jnp.zeros(num_actions)
    a_t_vector = a_t_vector.at[a_t].set(1)

    # n-step transition, letting s_tm1 = s_tmn, and a_tm1 = a_tmn.
    return Transition_SF(
        s_tm1=transitions[0].s_tm1,
        a_tm1=transitions[0].a_tm1,
        a_tm1_vector=a_tm1_vector,
        r_t=r_t,
        discount_t=discount_t,
        s_t=transitions[-1].s_tm1,
        a_t_vector=a_t_vector,
        task=transitions[-1].task,
    )


def build_n_step_transition_task_one_hot_action(transitions):
    """Builds a single n-step transition from n 1-step transitions."""
    r_t = 0.0
    discount_t = 1.0
    for transition in transitions:
        r_t += discount_t * transition.r_t
        discount_t *= transition.discount_t

    # n-step transition, letting s_tm1 = s_tmn, and a_tm1 = a_tmn.
    return Transition_Task_OneHotAction(
        s_tm1=transitions[0].s_tm1,
        a_tm1=transitions[0].a_tm1,
        a_tm1_vector=transitions[0].a_tm1_vector,
        r_t=r_t,
        discount_t=discount_t,
        s_t=transitions[-1].s_tm1,
        task=transitions[-1].task,
    )
