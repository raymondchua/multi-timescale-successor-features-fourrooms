import jax
import rlax
from typing import Unpack, Callable, Tuple, Any, Text, Mapping, Union
import chex
import distrax
import optax
import dm_env
import pickle
from flax.core import frozen_dict

from .DQN_agent import (
    DQNAgent,
    DQNAgentKwargs,
    Array,
    Params,
    Network,
    PRNGKey,
    Transition,
    Action,
)

from network import QNetworkMinatar, QNetOutputs, DQNetwork_named
import jax.numpy as jnp
from functools import partial
from helpers import (
    cbp_generate_and_test_dqn_head, init_cbp_state
)
from absl import logging
from pathlib import Path
from collections import OrderedDict


# Batch variant of q_learning.
_batch_q_learning = jax.vmap(rlax.q_learning)
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)


class DQNCBPAgentKwargs(DQNAgentKwargs):
    """Keyword arguments for DQN agent."""
    dead_eps: float
    maturity_threshold: int
    bottom_frac: float
    ema_decay: float
    replacement_rate: float
    enable_cbp_after: int


class DQNCBPAgent(DQNAgent):
    def __init__(self, **kwargs: Unpack[DQNCBPAgentKwargs]):
        super().__init__(**kwargs)

        self._kwargs = kwargs
        self._network = self.get_network_fn()
        self._network_eval = self._network

        self._online_params = self._network.init(
            self._network_rng_key, self._sample_network_input_extended
        )["params"]
        self._target_params = self._online_params
        self._opt_state = self._optimizer.init(self._online_params)

        self._dead_eps = self._kwargs["dead_eps"]
        self._maturity_threshold = self._kwargs["maturity_threshold"]
        self._bottom_frac = self._kwargs["bottom_frac"]
        self._ema_decay = self._kwargs["ema_decay"]
        self._replacement_rate = self._kwargs["replacement_rate"]
        self._enable_cbp_after = self._kwargs["enable_cbp_after"]
        self._cbp_state = init_cbp_state(self._kwargs["feature_dim"], self._kwargs["hidden_dim"])

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
        )

        online_output = network.apply(
            {"params": online_params},
            transitions.s_tm1,
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
        td_error_scaler_sigma: Array
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
        )

        target_output = network.apply(
            {"params": target_params},
            transitions.s_t,
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

    def get_network_fn(self) -> Network:
        """Returns a function that computes the deep neural network output."""

        if self._kwargs["env_type"] == "minatar":

            return QNetworkMinatar(
                num_actions=self._kwargs["action_shape"],
                feature_dim=self._kwargs["feature_dim"],
            )
        else:
            return DQNetwork_named(
                feature_dim=self._kwargs["feature_dim"],
                hidden_dim=self._kwargs["hidden_dim"],
                num_actions=self._kwargs["action_shape"],
            )

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "network",
            "loss_fn",
            "optimizer",
            "batch_size",
            "grad_error_bound",
            "log_grads",
            "replacement_rate",
            "ema_decay",
            "maturity_threshold",
            "dead_eps",
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
        enable_after_steps: int,
        replacement_rate: float,
        ema_decay: float,
        maturity_threshold: int,
        dead_eps: float,
        step: int,
        cbp_state: Any,
        log_grads: bool = False,
    ):
        """Computes learning update from batch of replay transitions."""
        rng_key, apply_key, target_key, loss_key, policy_key = jax.random.split(
            rng_key, 5
        )

        target_output = network.apply(
            {"params": target_params},
            transitions.s_t,
        )

        online_output = network.apply(
            {"params": online_params},
            transitions.s_tm1,
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
        )

        updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state, params=online_params)
        new_online_params = optax.apply_updates(online_params, updates)

        def do_cbp(args):
            rng_key, params, cbp_state = args
            rng_key, cbp_key = jax.random.split(rng_key)

            out, hs_new = network.apply({"params": params}, transitions.s_tm1, return_hs=True)
            cbp_key, new_params, new_cbp_state, debug = cbp_generate_and_test_dqn_head(
                cbp_key,
                params,
                cbp_state,
                hs=hs_new,
                dead_eps=dead_eps,
                rho=replacement_rate,
                eta=ema_decay,
                maturity=maturity_threshold,
            )

            return rng_key, new_params, new_cbp_state, debug

        def cbp_zero_debug():
            i32 = lambda: jnp.array(0, dtype=jnp.int32)
            f32 = lambda: jnp.array(0.0, dtype=jnp.float32)

            return {
                "replaced_rep": i32(),
                "replaced_hid": i32(),
                "replaced_total": i32(),
                "eligible_rep": i32(),
                "eligible_hid": i32(),
                "k_replace_rep": i32(),
                "k_replace_hid": i32(),
                "rho_n_out_rep": f32(),
                "rho_n_out_hid": f32(),
                "dead_rep": i32(),
                "dead_hid": i32(),
                "pool_count_rep": i32(),
                "pool_count_hid": i32(),
            }

        def skip_cbp(args):
            key, params, cbp_state = args
            cbp_debug = cbp_zero_debug()
            return key, params, cbp_state, cbp_debug

        rng_key, new_online_params, cbp_state, cbp_debug = jax.lax.cond(
            step >= enable_after_steps,
            do_cbp,
            skip_cbp,
            (rng_key, new_online_params, cbp_state),
        )

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
            cbp_state,
            cbp_debug
        )

    @staticmethod
    @partial(jax.jit, static_argnames="network")
    def _select_action(
        rng_key: PRNGKey,
        network_params: Params,
        s_t: Array,
        exploration_epsilon: float,
        network: Network,
    ) -> Tuple[PRNGKey, Array, Array]:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
        q_t = network.apply(
            {"params": network_params},
            s_t[None, ...],
        ).q_1[0]
        a_t = distrax.EpsilonGreedy(q_t, exploration_epsilon).sample(seed=policy_key)
        v_t = jnp.max(q_t, axis=-1)
        return rng_key, a_t, v_t

    def step(
        self,
        timestep: dm_env.TimeStep,
        time_to_learn: bool = False,
        action: int = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Performs a step in the environment and store the transition in replay.
        Parameters
        ----------
        timestep
        time_to_learn
        action: int - If action is provided, such as from a pre-trained policy, we use that action instead of sampling.
        kwargs

        Returns
        -------

        """
        self._frame_t += 1
        self._frame_t_exploration += 1
        metrics = dict()
        timestep = self._preprocessor(timestep)

        if timestep is None:  # Repeat action.
            if action is None:
                action = self._action
            else:
                self._action = action
        else:
            if action is None:
                action = self._action = self._act(timestep)

            else:
                self._action = action
                self._act(
                    timestep
                )  # just to store the state value, action is not being used

            for transition in self._transition_accumulator.step(timestep, action):

                if self._kwargs["use_priority_replay"]:
                    self._replay.add(transition, priority=self._max_seen_priority)

                else:
                    self._replay.add(transition)

        if time_to_learn:
            learn_metrics = self._learn()

            # best to log the wandb metric here for the training loss
            metrics.update(learn_metrics)

        # update critic target
        if self._frame_t % self._kwargs["target_network_update_period"] == 0:
            if self._kwargs["use_soft_target"]:
                self._target_params = jax.tree_util.tree_map(
                    lambda x, y: self._kwargs["critic_target_tau"] * x
                    + ((1 - self._kwargs["critic_target_tau"]) * y),
                    self._online_params,
                    self._target_params,
                )
            else:
                self._target_params = self._online_params

        metrics["action"] = action
        metrics["exploration_epsilon"] = self.exploration_epsilon
        return metrics

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
            self._cbp_state,
            cbp_debug,
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
            enable_after_steps=self._enable_cbp_after,
            replacement_rate=self._replacement_rate,
            ema_decay=self._ema_decay,
            maturity_threshold=self._maturity_threshold,
            step=self._frame_t,
            cbp_state=self._cbp_state,
            dead_eps=self._dead_eps,
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
        metrics = self.log_cbp(metrics, cbp_debug)

        replaced_total = self._to_py_scalar(cbp_debug["replaced_total"])

        # if replaced_total > 0:
        #     print("replaced_rep", self._to_py_scalar(cbp_debug["replaced_rep"]))
        #     print("replaced_hid", self._to_py_scalar(cbp_debug["replaced_hid"]))
        #     print("replaced_total", replaced_total)
        #     print("eligible_rep", self._to_py_scalar(cbp_debug["eligible_rep"]))
        #     print("eligible_hid", self._to_py_scalar(cbp_debug["eligible_hid"]))
        #     print("k_replace_rep", self._to_py_scalar(cbp_debug["k_replace_rep"]))
        #     print("k_replace_hid", self._to_py_scalar(cbp_debug["k_replace_hid"]))
        #     print("rho_n_out_rep", self._to_py_scalar(cbp_debug["rho_n_out_rep"]))
        #     print("rho_n_out_hid", self._to_py_scalar(cbp_debug["rho_n_out_hid"]))
        #     print("dead_rep", self._to_py_scalar(cbp_debug["dead_rep"]))
        #     print("dead_hid", self._to_py_scalar(cbp_debug["dead_hid"]))
        #     print("pool_count_rep", self._to_py_scalar(cbp_debug["pool_count_rep"]))
        #     print("pool_count_hid", self._to_py_scalar(cbp_debug["pool_count_hid"]))

        return metrics

    def _to_py_scalar(self, x):
        # Works for jnp scalars / numpy scalars / python numbers
        try:
            return x.item()
        except Exception:
            return x

    def log_cbp(self, metrics: dict, cbp_debug, *, prefix: str = ""):
        """
        Copy cbp_debug entries into metrics as python scalars.
        """
        for k, v in cbp_debug.items():
            name = k
            if prefix:
                name = f"{prefix}{name}"

            # if the value is a tuple (like for dead_counts), we convert each element to a scalar and store if separately
            # as dead_rep, dead_hid, dead_total
            if isinstance(v, tuple):
                for i, sub_v in enumerate(v):
                    if i == 0:
                        sub_name = f"{name}_rep"
                    elif i == 1:
                        sub_name = f"{name}_hid"
                    elif i == 2:
                        sub_name = f"{name}_total"
                    else:
                        sub_name = f"{name}_{i}"
                    metrics[sub_name] = self._to_py_scalar(sub_v)
            else:
                metrics[name] = self._to_py_scalar(v)

        return metrics