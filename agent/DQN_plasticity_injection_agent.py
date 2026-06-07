from flax.core import freeze, unfreeze, FrozenDict
from typing_extensions import Unpack

import chex
import rlax
import jax
import jax.numpy as jnp
import jax.random as jr
import distrax
import optax
import dm_env

from absl import logging
from typing import Unpack, Callable, Tuple, Any, Text, Mapping, Union
from functools import partial

from .DQN_agent import (
    Array,
    DQNAgent,
    DQNAgentKwargs,
    Network,
    PRNGKey,
    Params,
    Transition,
    Action,
)

from helpers import (
    processors,
    ReplayStructure,
    TransitionAccumulator,
    NStepTransitionAccumulator,
)

from flax.traverse_util import flatten_dict
from network import QNetOutputs, DQNetworkPI

# Batch variant of q_learning.
_batch_q_learning = jax.vmap(rlax.q_learning)
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)

class DQN_Plasticity_Injection_Agent(DQNAgent):
    def __init__(
        self,
        **kwargs: Unpack[DQNAgentKwargs],
    ):
        super().__init__(**kwargs)
        self._pi_enabled = False
        self._network = self.get_network_fn()
        self._network_eval = self._network
        self._online_params = self._network.init(
            self._network_rng_key, self._sample_network_input_extended
        )["params"]
        print(self._online_params.keys())
        self._online_params["head_new"] = self._online_params["head_copy"]
        kc, kn = self._online_params["head_copy"]["kernel"], self._online_params["head_new"]["kernel"]
        bc, bn = self._online_params["head_copy"]["bias"], self._online_params["head_new"]["bias"]
        assert jnp.array_equal(kc, kn) and jnp.array_equal(bc, bn)
        self._target_params = self._online_params
        self._opt_state = self._optimizer.init(self._online_params)

    def get_network_fn(self) -> Network:
        """Returns a function that computes the deep neural network output."""

        return DQNetworkPI(
            feature_dim=self._kwargs["feature_dim"],
            hidden_dim=self._kwargs["hidden_dim"],
            num_actions=self._kwargs["action_shape"],
        )

    def pi_enabled(self):
        self._pi_enabled = True
        logging.info("Plasticity Injection enabled")

    @staticmethod
    def loss_fn(
        online_params: Params,
        target_params: Params,
        transitions: Transition,
        rng_key: PRNGKey,
        network: Network,
        grad_error_bound: float,
        batch_size: int,
        weights: Array,
        td_error_scaler_sigma: Array,
        inject: bool,
    ) -> [float, Array]:
        """

        Parameters
        ----------
        online_params: Params
        target_params: Params
        transitions: Transition
        network: Network
        weights: Array - Importance sampling weights for prioritized replay
        rng_key: PRNGKey
        grad_error_bound: float
        batch_size: int
        td_error_scaler_sigma: Array

        Returns
        -------
        critic_loss: Array
        td_errors: Array - TD errors for prioritized replay

        """
        (
            _,
            apply_key,
            target_key,
        ) = jax.random.split(rng_key, 3)

        target_output = network.apply(
            {"params": target_params},
            transitions.s_t,
            inject=inject,
        )

        online_output = network.apply(
            {"params": online_params},
            transitions.s_tm1,
            inject=inject,
        )
        target_Q1 = target_output.q_1

        q_1 = online_output.q_1

        # have to do it like how it is done in dqnzoo as we need the td_errors
        td_errors = _batch_q_learning(
            q_1,
            transitions.a_tm1,
            transitions.r_t,
            transitions.discount_t,
            target_Q1,
        )
        td_errors = rlax.clip_gradient(td_errors, -grad_error_bound, grad_error_bound)
        losses = rlax.l2_loss(td_errors)
        chex.assert_shape((losses, weights), (batch_size,))
        # This is not the same as using a huber loss and multiplying by weights.
        loss = jnp.mean(losses * weights)
        loss /= td_error_scaler_sigma
        return loss, td_errors

    @staticmethod
    def double_q_loss_fn(
        online_params: Params,
        target_params: Params,
        transitions: Transition,
        rng_key: PRNGKey,
        network: Network,
        grad_error_bound: float,
        batch_size: int,
        weights: Array,
        td_error_scaler_sigma: Array,
        inject: bool,
    ) -> [float, Array]:
        """

        Parameters
        ----------
        online_params: Params
        target_params: Params
        transitions: Transition
        network: Network
        weights: Array - Importance sampling weights for prioritized replay
        rng_key: PRNGKey
        grad_error_bound: float
        batch_size: int
        td_error_scaler_sigma: Array

        Returns
        -------
        critic_loss: Array
        td_errors: Array - TD errors for prioritized replay

        """
        _, *apply_keys = jax.random.split(rng_key, 4)

        online_tm1_output = network.apply({"params": online_params}, transitions.s_tm1)
        online_t_output = network.apply(
            {"params": online_params},
            transitions.s_t,
            inject=inject,
        )

        target_output = network.apply(
            {"params": target_params},
            transitions.s_t,
            inject=inject,
        )

        q_target_t = target_output.q_1

        q_tm1 = online_tm1_output.q_1
        q_t = online_t_output.q_1

        # have to do it like how it is done in dqnzoo as we need the td_errors
        td_errors = _batch_double_q_learning(
            q_tm1,
            transitions.a_tm1,
            transitions.r_t,
            transitions.discount_t,
            q_target_t,
            q_t,
        )

        td_errors = rlax.clip_gradient(td_errors, -grad_error_bound, grad_error_bound)
        losses = rlax.l2_loss(td_errors)
        chex.assert_shape((losses, weights), (batch_size,))
        # This is not the same as using a huber loss and multiplying by weights.
        loss = jnp.mean(losses * weights)
        loss /= td_error_scaler_sigma
        return loss, td_errors

    @staticmethod
    @partial(jax.jit, static_argnames="network")
    def _get_output(
        rng_key: PRNGKey,
        online_critic_sf_params: Params,
        s_t: Array,
        network: Network,
        inject: bool,
    ) -> QNetOutputs:
        """Forward pass through the network for inference. No gradient."""
        _, apply_key = jax.random.split(rng_key)
        output = network.apply(
            {"params": online_critic_sf_params},
            s_t[None, ...],
            inject=inject,
        )

        return jax.lax.stop_gradient(output)

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "network",
            "loss_fn",
            "optimizer",
            "batch_size",
            "grad_error_bound",
            "inject",
            "log_grads",
        ],
    )
    def _update(
            rng_key: PRNGKey,
            opt_state: optax.OptState,
            online_params: Params,
            target_params: Params,
            transitions: Transition,
            network: Network,
            loss_fn: Callable,
            optimizer: optax.GradientTransformation,
            td_error_scaler_sigma: Array,
            grad_error_bound: float,
            batch_size: int,
            weights: Array,
            inject: bool,
            log_grads: bool = False,
    ):
        """Computes learning update from batch of replay transitions."""
        rng_key, apply_key, target_key, loss_key, policy_key = jax.random.split(
            rng_key, 5
        )
        target_output = network.apply(
            {"params": target_params},
            transitions.s_t,
            inject=inject,
        )

        online_output = network.apply(
            {"params": online_params},
            transitions.s_tm1,
            inject=inject,
        )

        target_Q1 = jax.lax.stop_gradient(target_output.q_1)
        target_V = jnp.max(target_Q1, axis=-1)

        reward = jnp.squeeze(transitions.r_t)
        target_Q = reward + (transitions.discount_t * target_V)
        target_Q = jax.lax.stop_gradient(target_Q)

        q_1 = jax.lax.stop_gradient(online_output.q_1)

        (critic_loss, td_errors), d_loss_d_params = jax.value_and_grad(
            loss_fn, has_aux=True
        )(
            online_params,
            target_params,
            transitions,
            loss_key,
            network,
            grad_error_bound,
            batch_size,
            weights,
            td_error_scaler_sigma,
            inject,
        )

        updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
        new_online_params = optax.apply_updates(online_params, updates)

        if log_grads:
            grads = d_loss_d_params
        else:
            grads = None

        return (
            rng_key,
            new_opt_state,
            new_online_params,
            target_Q,
            q_1,
            critic_loss,
            td_errors,
            grads,
        )

    @staticmethod
    @partial(jax.jit, static_argnames="network")
    def _select_action(
        rng_key: PRNGKey,
        network_params: Params,
        s_t: Array,
        exploration_epsilon: float,
        network: Network,
        inject: bool,
    ) -> Tuple[PRNGKey, Array, Array]:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
        q_t = network.apply(
            {"params": network_params},
            s_t[None, ...],
            inject=inject,
        ).q_1[0]
        a_t = distrax.EpsilonGreedy(q_t, exploration_epsilon).sample(seed=policy_key)
        v_t = jnp.max(q_t, axis=-1)
        return rng_key, a_t, v_t

    def _act(self, timestep: dm_env.TimeStep) -> Action:
        """Selects action given timestep, according to epsilon-greedy policy."""
        s_t = timestep.observation
        self._rng_key, a_t, v_t = self._select_action(
            rng_key=self._rng_key,
            network_params=self._online_params,
            s_t=s_t,
            exploration_epsilon=self.exploration_epsilon,
            network=self._network,
            inject=self._pi_enabled,
        )

        a_t, v_t = jax.device_get((a_t, v_t))
        self._statistics["state_value"] = v_t
        return Action(a_t)

    def _learn(self) -> dict[str, float]:
        """Samples a batch of transitions from replay and learns from it."""
        logging.log_first_n(logging.INFO, "Begin learning", 1)
        indices = None
        # if we are to use priority replay, then use the loss_fn and use the weights from the samples
        # else use the normal loss_fn and use the w_critic from the kwargs
        if self._kwargs["use_priority_replay"]:
            transitions, indices, weights = self._replay.sample(
                self._kwargs["batch_size"]
            )

        else:
            transitions = self._replay.sample(self._kwargs["batch_size"])
            weights = jnp.ones(self._kwargs["batch_size"])

        if self._kwargs["use_double_q"]:
            loss_fn = self.double_q_loss_fn
        else:
            loss_fn = self.loss_fn

        reward_from_env = transitions.r_t

        metrics = dict()

        (
            self._train_rng_key,
            self._opt_state,
            self._online_params,
            target_Q,
            q_1,
            loss,
            td_errors,
            grads,
        ) = self._update(
            rng_key=self._train_rng_key,
            opt_state=self._opt_state,
            online_params=self._online_params,
            target_params=self._target_params,
            transitions=transitions,
            network=self._network,
            loss_fn=loss_fn,
            optimizer=self._optimizer,
            grad_error_bound=self._kwargs["grad_error_bound"],
            batch_size=self._kwargs["batch_size"],
            weights=weights,
            log_grads=self._kwargs["log_grads"],
            td_error_scaler_sigma=self.td_sigma,
            inject=self._pi_enabled,
        )

        if self._kwargs["log_grads"]:
            assert grads is not None
            metrics = self.log_grads_norm(
                metrics=metrics, gradients=grads, name="critic"
            )

        # update the priorities if we are using priority replay
        if self._kwargs["use_priority_replay"]:
            chex.assert_equal_shape((weights, td_errors))
            self.update_priorities(indices=indices, td_errors=td_errors)

        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["q_1"] = q_1.mean().item()
        metrics["critic_loss"] = loss.item()
        metrics["extr_reward"] = reward_from_env.mean().item()
        metrics["td_error_scaler_sigma"] = self.td_sigma.item()

        return metrics

    def get_state(self) -> dict[str, Any]:
        """Retrieves agent state as a dictionary (e.g. for serialization)."""
        state = {
            "rng_key": self._rng_key,
            "frame_t": self._frame_t,
            "opt_state": self._opt_state,
            "online_params": self._online_params,
            "target_params": self._target_params,
            "replay": self._replay.get_state(),
            "pi_enabled": self._pi_enabled,
        }
        return state

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets agent state from a (potentially de-serialized) dictionary."""
        self._rng_key = state["rng_key"]
        self._frame_t = state["frame_t"]
        self._opt_state = jax.device_put(state["opt_state"])
        self._online_params = jax.device_put(state["online_params"])
        self._target_params = jax.device_put(state["target_params"])
        self._replay.set_state(state["replay"])
        self._pi_enabled = state["pi_enabled"]

    def get_rep_and_val(self, timestep: dm_env.TimeStep):
        timestep = self._preprocessor(timestep)
        s_t = timestep.observation
        output = self._get_output(
            rng_key=self._eval_rng_key,
            online_critic_sf_params=self._online_params,
            s_t=s_t,
            network=self._network,
            inject=self._pi_enabled,
        )
        qval_t = output.q_1[0]
        obs_t = output.obs_rep[0]
        v_t = jnp.max(qval_t, axis=-1)

        return (
            v_t,
            obs_t,
            qval_t,
        )


