from .BaseAgent import *

import jax
import dm_env
import distrax

from typing import Any, Iterable, Mapping, Optional, Text, Tuple, Union, List
import helpers.processors as processors
from absl import logging
import chex
import jax.numpy as jnp

from .simple_sf_agent import (
    Action,
    SimpleSFAgent,
    SimpleSFAgentKwargs,
    Transition_Task,
    Params,
    Array,
    PRNGKey,
    Network,
)

action_mapping_minigrid = {
    "left": 0,  # Turn left
    "right": 1,  # Turn right
    "forward": 2,  # Move forward
    "pickup": 3,  # Pick up an object
    "drop": 4,  # Drop an object
    "toggle": 5,  # Toggle/activate an object
    "done": 6,  # Done completing task
}

action_mapping_miniworld = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "move_back": 3,
    "pickup": 4,
    "drop": 5,
    "toggle": 6,
    "done": 7,
}


class RandomPolicyBiasForwardAgent(SimpleSFAgent):
    """Agent that acts with a given set of Q-network parameters and epsilon.

    Network parameters are set on the actor. The actor can be serialized,
    ensuring determinism of execution (e.g. when checkpointing).
    """

    def __init__(
        self,
        **kwargs: Unpack[SimpleSFAgentKwargs],
    ):
        super().__init__(**kwargs)
        self._consolidation = self._kwargs["consolidation"]
        self._env_type = self._kwargs["env_type"]

    def _act(self, timestep: dm_env.TimeStep, task: Array) -> Action:
        """Selects action given timestep, according to epsilon-greedy policy."""
        s_t = timestep.observation
        task_no_grad = jax.lax.stop_gradient(task)
        self._train_rng_key, a_t, v_t = self._select_action(
            self._train_rng_key,
            self._online_params,
            s_t,
            self.exploration_epsilon,
            task_no_grad,
            self._network,
            self._env_type,
        )
        a_t, v_t = jax.device_get((a_t, v_t))
        self._statistics["state_value"] = v_t
        return Action(a_t)

    def act_eval(
        self,
        timestep: dm_env.TimeStep,
        task: Array = None,
    ) -> Mapping[Text, Any]:
        """Selects action given timestep and with no learning involved."""
        timestep = self._preprocessor(timestep)
        metric = dict()

        if timestep is None:  # Repeat action.
            metric["action"] = self._action
            return metric

        s_t = timestep.observation
        task_no_grad = jax.lax.stop_gradient(task)
        self._rng_key, a_t, v_t = self._select_action(
            self._rng_key,
            self._online_params,
            s_t,
            self.eval_exploration_epsilon,
            task_no_grad,
            self._network,
            self._env_type,
        )
        self._rng_key, pred_reward = self._get_predicted_reward(
            rng_key=self._rng_key,
            online_params=self._online_params,
            s_t=s_t,
            task=task_no_grad,
            network=self._network,
        )

        self._action, pred_reward, v_t = jax.device_get((a_t, pred_reward, v_t))
        metric["action"] = self._action
        metric["pred_reward"] = pred_reward
        metric["v_t"] = v_t
        metric["exploration_epsilon"] = self.exploration_epsilon
        return metric

    @staticmethod
    @partial(jax.jit, static_argnames=["network", "env_type"])
    def _select_action(
        rng_key: PRNGKey,
        online_params: Params,
        s_t: Array,
        exploration_epsilon: float,
        task: Array,
        network: Network,
        env_type: str,
    ) -> tuple[PRNGKey, Array, Array]:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        rng_key, sf_critic_key, policy_key = jax.random.split(rng_key, 3)
        q_t = network.apply(
            {"params": online_params},
            s_t[None, ...],
            task[None, ...],
        ).q_1[0]

        if env_type == "minigrid":
            action_space = jnp.zeros(len(action_mapping_minigrid))
            action_space = action_space.at[action_mapping_minigrid["forward"]].set(0.6)
            action_space = action_space.at[action_mapping_minigrid["left"]].set(0.15)
            action_space = action_space.at[action_mapping_minigrid["right"]].set(0.15)
            action_space = action_space.at[action_mapping_minigrid["done"]].set(0.1)

        elif env_type == "miniworld":
            action_space = jnp.zeros(len(action_mapping_miniworld))
            action_space = action_space.at[action_mapping_miniworld["forward"]].set(0.6)
            action_space = action_space.at[action_mapping_miniworld["left"]].set(0.15)
            action_space = action_space.at[action_mapping_miniworld["right"]].set(0.15)
            action_space = action_space.at[action_mapping_miniworld["done"]].set(0.1)

        else:
            raise ValueError(
                f"Unknown environment type: {env_type}. "
                "Supported types are 'minigrid' and 'miniworld'."
            )
        v_t = jnp.max(q_t, axis=-1)
        a_t = jax.random.choice(
            policy_key,
            jnp.arange(len(action_space)),
            p=action_space,
            shape=(),
            replace=False,
        )
        return rng_key, a_t, v_t

    def act_random_bias_forward(
        self,
    ) -> dict[Text, Any]:
        """Selects action given timestep and potentially learns."""
        metric = dict()
        # get the action space for the environment type. Determine which action is move forward, turn left, turn right
        # and do nothing. Set the probability for the actions such that they have the probability values of 0.6, 0.15,
        # 0.15, and 0.1, respectively.
        if self._env_type == "minigrid":
            action_space = np.zeros(len(action_mapping_minigrid))
            action_space[action_mapping_minigrid["forward"]] = 0.6
            action_space[action_mapping_minigrid["left"]] = 0.15
            action_space[action_mapping_minigrid["right"]] = 0.15
            action_space[action_mapping_minigrid["done"]] = 0.1

        elif self._env_type == "miniworld":
            action_space = np.zeros(len(action_mapping_miniworld))
            action_space[action_mapping_miniworld["forward"]] = 0.6
            action_space[action_mapping_miniworld["left"]] = 0.15
            action_space[action_mapping_miniworld["right"]] = 0.15
            action_space[action_mapping_miniworld["done"]] = 0.1

        else:
            raise ValueError(
                f"Unknown environment type: {self._env_type}. "
                "Supported types are 'minigrid' and 'miniworld'."
            )

        # randomly select an action from the action space
        self._action = np.random.choice(np.arange(len(action_space)), p=action_space)

        metric["action"] = self._action
        return metric

    @property
    def has_attention_mechanism(self) -> bool:
        return self._kwargs["has_attention_mechanism"]

    @property
    def eval_exploration_epsilon(self) -> float:
        """Returns epsilon value currently used by (eps-greedy) behavior policy. In this agent we return zero as it does
        not use epsilon-greedy exploration."""
        return self._kwargs["eval_exploration_epsilon"]

    @property
    def consolidation(self) -> bool:
        return self._consolidation
