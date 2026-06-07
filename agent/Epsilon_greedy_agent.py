from .BaseAgent import *

import jax
import dm_env
import distrax

from typing import Any, Iterable, Mapping, Optional, Text, Tuple, Union, List
import helpers.processors as processors
from absl import logging
import chex
import jax.numpy as jnp

Action = int
Array = chex.Array
PRNGKey = jnp.ndarray  # A size 2 array.
from flax.core import freeze, FrozenDict


class EpsilonGreedyActor:
    """Agent that acts with a given set of Q-network parameters and epsilon.

    Network parameters are set on the actor. The actor can be serialized,
    ensuring determinism of execution (e.g. when checkpointing).
    """

    def __init__(self, cfg: Any, **kwargs):
        self._cfg = cfg
        self._rng_key = None
        self._action = None
        self._network = None
        self._network_params = None
        self._preprocessor = None
        self._preprocessors_list = None
        self._exploration_epsilon = self._cfg.agent.eval_exploration_epsilon
        self._use_preprocessor = (self._cfg.agent.use_preprocessor,)
        self._env_type = self._cfg.domain.env_type
        self._environment_width = self._cfg.domain.environment_width
        self._environment_height = self._cfg.domain.environment_height
        self._discount = self._cfg.domain.discount
        self._pi_enabled = False  # set True when params/network are PI
        self._online_frozen = None  # FrozenDict for 'frozen' vars (optional)

    def _vars_online(self) -> FrozenDict:
        """Return a variables dict suitable for network.apply.

        Accepts either:
          - raw params: {"...layer...": {...}}, or
          - full variables: {"params": ..., "frozen": ...}.
        Ensures we add 'frozen' if PI is enabled.
        """
        p = self._network_params

        # Case A: caller already provided a full variables dict
        if isinstance(p, (dict, FrozenDict)) and "params" in p:
            params_tree = p["params"]
            frozen_tree = p.get("frozen", None)
            # If we have our own frozen tree (from train agent), prefer it
            if getattr(self, "_pi_enabled", False) and getattr(self, "_online_frozen", None) is not None:
                frozen_tree = self._online_frozen
            out = {"params": params_tree}
            if frozen_tree is not None:
                out["frozen"] = frozen_tree
            return freeze(out)

        # Case B: raw params tree; wrap it
        out = {"params": p}
        if getattr(self, "_pi_enabled", False) and getattr(self, "_online_frozen", None) is not None:
            out["frozen"] = self._online_frozen
        return freeze(out)

    def _is_pi_params(self, params_tree) -> bool:
        try:
            if isinstance(params_tree, (dict, FrozenDict)) and "params" in params_tree:
                keys = list(params_tree["params"].keys())
            else:
                keys = list(params_tree.keys())
        except Exception:
            return False
        return ("pi_head" in keys) or ("PIHead" in keys)

    def maybe_switch_to_pi(self):
        """If params look like PI, require a PI network and mark PI enabled."""
        if self._is_pi_params(self._network_params):
            # The caller should set self._network to DQNetworkPI; assert or fallback.
            from network import DQNetworkPI  # import here to avoid cycles
            if not isinstance(self._network, DQNetworkPI):
                # If your eval wiring doesn't pass the model, construct a default PI model.
                self._network = DQNetworkPI(
                    feature_dim=self._cfg.agent.feature_dim,
                    hidden_dim=self._cfg.agent.hidden_dim,
                    num_actions=self._cfg.agent.action_shape,
                )
            self._pi_enabled = True

    @staticmethod
    @partial(jax.jit, static_argnames=["network", "exploration_epsilon"])
    def _select_action(rng_key, network, vars_online, s_t, task, exploration_epsilon, mask):
        assert task is not None, "task is None"

        """Samples action from eps-greedy policy wrt Q-values at given state."""
        rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)

        if mask is None:
            q_t = network.apply(
                vars_online,
                s_t[None, ...],
                task[None, ...],
            ).q_1[0]

        else:
            q_t = network.apply(
                vars_online,
                s_t[None, ...],
                task[None, ...],
                mask[0, :, :, :],
            ).q_1[0]

        q_t = jnp.squeeze(q_t)
        q_t = jax.lax.stop_gradient(q_t)
        a_t = distrax.EpsilonGreedy(
            preferences=q_t, epsilon=exploration_epsilon
        ).sample(seed=policy_key)
        v_t = jnp.max(q_t, axis=-1)
        return rng_key, a_t, v_t

    @staticmethod
    @partial(jax.jit, static_argnames=["network", "exploration_epsilon"])
    def _select_action_without_meta(rng_key, network, vars_online, s_t, exploration_epsilon: float,):
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
        q_t = network.apply(
            vars_online,
            s_t[None, ...],
        ).q_1[0]

        # if agent has beakers, then q_t uses the first beaker
        if len(q_t.shape) > 1:
            q_t = q_t[0]

        q_t = jnp.squeeze(q_t)
        q_t = jax.lax.stop_gradient(q_t)

        a_t = distrax.EpsilonGreedy(
            preferences=q_t, epsilon=exploration_epsilon
        ).sample(seed=policy_key)
        v_t = jnp.max(q_t, axis=-1)
        return rng_key, a_t, v_t

    @staticmethod
    @partial(jax.jit, static_argnames=["network"])
    def _get_predicted_reward(rng_key, network, vars_online, s_t, task):
        rng_key, apply_key = jax.random.split(rng_key)

        if len(task.shape) == 1:
            task = jnp.expand_dims(task, axis=0)

        basis_features_t = network.apply(
            vars_online,
            s_t[None, ...],
            task,
        ).basis_features

        pred_reward = jnp.dot(jnp.squeeze(basis_features_t), jnp.squeeze(task))
        pred_reward = jnp.squeeze(pred_reward)
        return rng_key, jax.lax.stop_gradient(pred_reward)

    def act_eval(
        self,
        timestep: dm_env.TimeStep,
        task: Array = None,
        mask: Array = None,
        recall_gain: Array = None,
    ) -> dict[Text, Any]:
        """Selects action given timestep and potentially learns."""
        timestep = self._preprocessor(timestep)
        metric = dict()

        if timestep is None:  # Repeat action.
            metric["action"] = self._action
            return metric

        s_t = timestep.observation
        task_no_grad = jax.lax.stop_gradient(task)
        self._rng_key, a_t, v_t = self._select_action(
            self._rng_key, self._network, self._vars_online(), s_t, task_no_grad, self._exploration_epsilon, mask
        )
        self._rng_key, pred_reward = self._get_predicted_reward(
            self._rng_key, self._network, self._vars_online(), s_t, task_no_grad,
        )
        self._action, pred_reward, v_t = jax.device_get((a_t, pred_reward, v_t))

        metric["action"] = self._action
        metric["pred_reward"] = float(pred_reward)
        metric["v_t"] = jnp.squeeze(v_t)
        metric["exploration_epsilon"] = self._exploration_epsilon
        return metric

    def act_eval_without_meta(
        self,
        timestep: dm_env.TimeStep,
    ) -> dict[Text, Any]:
        """Selects action given timestep and potentially learns."""
        timestep = self._preprocessor(timestep)
        metric = dict()

        if timestep is None:  # Repeat action.
            metric["action"] = self._action
            return metric

        s_t = timestep.observation
        self._rng_key, a_t, v_t = self._select_action_without_meta(
            self._rng_key, self._network, self._vars_online(), s_t, self._exploration_epsilon,
        )
        self._action, v_t = jax.device_get((a_t, v_t))
        metric["action"] = self._action
        metric["v_t"] = jnp.squeeze(v_t)
        metric["exploration_epsilon"] = self._exploration_epsilon
        return metric

    def step(
        self, timestep: dm_env.TimeStep, meta: Array = None, mask: Array = None, recall_gain: Array = None,
    ) -> dict[Text, Any]:
        """Selects action given a timestep."""

        if meta is not None:
            return self.act_eval(timestep, meta, mask, recall_gain)

        else:
            return self.act_eval_without_meta(timestep)

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """
        processors.reset(self._preprocessor)
        self._action = None

    def get_state(self) -> Mapping[Text, Any]:
        out = {
            "rng_key": self._rng_key,
            "network_params": self.network_params,
        }
        if self._pi_enabled and (self._online_frozen is not None):
            out["pi_enabled"] = True
            out["online_frozen"] = self._online_frozen
        return out

    def set_state(self, state: Mapping[Text, Any]) -> None:
        self._rng_key = state["rng_key"]
        self.network_params = state["network_params"]
        # restore PI extras if present
        self._pi_enabled = bool(state.get("pi_enabled", False))
        if "online_frozen" in state:
            self._online_frozen = jax.device_put(state["online_frozen"])
        # ensure network matches the params
        self.maybe_switch_to_pi()

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {}

    @property
    def eval_rng_key(self) -> PRNGKey:
        return self._rng_key

    @eval_rng_key.setter
    def eval_rng_key(self, rng_key: PRNGKey) -> None:
        logging.log_first_n(logging.INFO, "RNG key for eval set", 1)
        self._rng_key = rng_key

    @property
    def network(self) -> Network:
        return self._network

    @network.setter
    def network(self, network: Network) -> None:
        logging.log_first_n(logging.INFO, "Network for eval set", 1)
        self._network = network

    @property
    def online_params(self) -> Params:
        return self._network_params

    @online_params.setter
    def online_params(self, network_params: Params) -> None:
        logging.log(logging.INFO, "Network params for epsilon greedy agent")

        # If a full variables dict was provided, unwrap and capture 'frozen'
        if isinstance(network_params, (dict, FrozenDict)) and "params" in network_params:
            self._network_params = network_params["params"]
            if "frozen" in network_params:
                self._online_frozen = network_params["frozen"]
                self._pi_enabled = True
        else:
            self._network_params = network_params

        self.maybe_switch_to_pi()

    def preprocessors(self, preprocessor: Callable):
        self._preprocessor = preprocessor

    def set_preprocessor(self, task_id: int = 0):
        self._preprocessor = self._preprocessors_list[task_id]

    @property
    def exploration_epsilon(self) -> float:
        return self._exploration_epsilon