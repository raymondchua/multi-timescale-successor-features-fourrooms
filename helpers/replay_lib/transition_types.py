from typing import Optional
import typing

import jax.numpy as jnp
import numpy as np


class Transition(typing.NamedTuple):
    s_tm1: Optional[jnp.ndarray]
    a_tm1: Optional[int]
    r_t: Optional[jnp.ndarray]
    discount_t: Optional[jnp.ndarray]
    s_t: Optional[jnp.ndarray]


class Transition_SF(typing.NamedTuple):
    s_tm1: Optional[jnp.ndarray]
    a_tm1: Optional[int]
    a_tm1_vector: Optional[jnp.ndarray]
    r_t: Optional[float]
    discount_t: Optional[float]
    s_t: Optional[jnp.ndarray]
    a_t_vector: Optional[jnp.ndarray]
    task: Optional[jnp.ndarray]


class Transition_Task(typing.NamedTuple):
    s_tm1: Optional[jnp.ndarray]
    a_tm1: Optional[int]
    r_t: Optional[jnp.ndarray]
    discount_t: Optional[jnp.ndarray]
    s_t: Optional[jnp.ndarray]
    task: Optional[jnp.ndarray]


class Transition_Task_Prev_Action(typing.NamedTuple):
    s_tm1: Optional[jnp.ndarray]
    a_tm1: Optional[int]
    a_tm2: Optional[int]
    r_t: Optional[jnp.ndarray]
    discount_t: Optional[jnp.ndarray]
    s_t: Optional[jnp.ndarray]
    task: Optional[jnp.ndarray]


class Transition_Task_OneHotAction(typing.NamedTuple):
    s_tm1: Optional[jnp.ndarray]
    a_tm1: Optional[int]
    a_tm1_vector: Optional[np.ndarray]
    r_t: Optional[float]
    discount_t: Optional[float]
    s_t: Optional[jnp.ndarray]
    task: Optional[jnp.ndarray]
