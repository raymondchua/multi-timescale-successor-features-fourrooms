import jax
import rlax
from typing import Unpack, Callable, Tuple, Any, Text, Mapping, Union
import chex
import distrax
import optax
import dm_env
import pickle

from dm_env import specs

from network import SFNetworkMinatar, SFNetOutputs, SFNetwork
from jax import random

import jax.numpy as jnp
import numpy as np

from .simple_sf_agent import (
    Array,
    SimpleSFAgent,
    SimpleSFAgentKwargs,
    Network,
    Params,
    PRNGKey,
    Transition_Task,
    Action,
)

from functools import partial
from helpers.ewc import (
    EWCState,
    compute_fisher_diagonal,
    update_fisher,
    consolidate_params,
    create_ewc_loss_fn,
)

from losses import reward_prediction_loss

from absl import logging
from pathlib import Path
from collections import OrderedDict

TaskParams = Mapping[str, Any]

_batch_reward_prediction_loss = jax.vmap(reward_prediction_loss)
_batch_q_learning = jax.vmap(rlax.q_learning)
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)


class SimpleSF_online_ewc_Agent(SimpleSFAgent):
    def __init__(
        self,
        **kwargs: Unpack[SimpleSFAgentKwargs],
    ):
        super().__init__(**kwargs)

        self._ewc_state = None
        self._ewc_lambda = self._kwargs.get("ewc_regularization", 0.0)
        self._ewc_gamma = self._kwargs.get("ewc_gamma", 0.9)
        self._fisher_update_interval = self._kwargs.get("fisher_update_interval", 10000)

    @staticmethod
    def loss_fn(
        online_params: Params,
        target_params: Params,
        transitions: Transition_Task,
        rng_key: PRNGKey,
        network: Network,
        grad_error_bound: float,
        batch_size: int,
        weights: Array,
        td_error_scaler_sigma: Array,
        ewc_state: EWCState,
        ewc_lambda: float,
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

        def task_loss_fn(params, *_):
            target_output = network.apply(
                {"params": target_params},
                transitions.s_t,
                transitions.task,
            )

            online_output = network.apply(
                {"params": online_params},
                transitions.s_tm1,
                transitions.task,
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
            return loss

        loss_with_ewc = create_ewc_loss_fn(task_loss_fn, ewc_state, ewc_lambda)
        return (
            loss_with_ewc(
                online_params,
                target_params,
                transitions,
                rng_key,
                network,
                grad_error_bound,
                batch_size,
                weights,
                td_error_scaler_sigma,
            ),
            None,
        )

    @staticmethod
    def double_q_loss_fn(
        online_params: Params,
        target_params: Params,
        transitions: Transition_Task,
        rng_key: PRNGKey,
        network: Network,
        grad_error_bound: float,
        batch_size: int,
        weights: Array,
        td_error_scaler_sigma: Array,
        ewc_state: EWCState,
        ewc_lambda: float,
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

        def task_loss_fn(params, *_):
            online_tm1_output = network.apply(
                {"params": online_params},
                transitions.s_tm1,
                transitions.task,
            )

            online_t_output = network.apply(
                {"params": online_params}, transitions.s_t, transitions.task
            )

            target_output = network.apply(
                {"params": target_params},
                transitions.s_t,
                transitions.task,
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
            return loss

        loss_with_ewc = create_ewc_loss_fn(task_loss_fn, ewc_state, ewc_lambda)
        return (
            loss_with_ewc(
                online_params,
                target_params,
                transitions,
                rng_key,
                network,
                grad_error_bound,
                batch_size,
                weights,
                td_error_scaler_sigma,
            ),
            None,
        )

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "network",
            "loss_fn",
            "optimizer_critic_sf",
            "grad_error_bound",
            "batch_size",
            "log_grads",
            "ewc_lambda",
        ],
    )
    def _update(
            rng_key: PRNGKey,
            opt_state: optax.OptState,
            online_params: Params,
            target_params: Params,
            transitions: Transition_Task,
            network: Network,
            loss_fn: Callable,
            optimizer_critic_sf: optax.GradientTransformation,
            td_error_scaler_sigma: Array,
            grad_error_bound: float,
            batch_size: int,
            weights: Array,
            ewc_state: EWCState,
            ewc_lambda: float,
            log_grads: bool = False,
    ):
        rng_key, apply_key, target_key, loss_key, policy_key = jax.random.split(
            rng_key, 5
        )

        target_output = network.apply(
            {"params": target_params},
            transitions.s_t,
            transitions.task,
        )

        online_output = network.apply(
            {"params": online_params},
            transitions.s_tm1,
            transitions.task,
        )

        target_Q1 = jax.lax.stop_gradient(target_output.q_1)
        target_V = jnp.max(target_Q1, axis=-1)
        reward = jnp.squeeze(transitions.r_t)
        target_Q = reward + (transitions.discount_t * target_V)
        target_Q = jax.lax.stop_gradient(target_Q)

        q_1 = jax.lax.stop_gradient(online_output.q_1)
        sf_1 = jax.lax.stop_gradient(online_output.sf)

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
            ewc_state,
            ewc_lambda,
        )

        updates, new_opt_state = optimizer_critic_sf.update(d_loss_d_params, opt_state)
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
            sf_1,
            critic_loss,
            td_errors,
            grads,
        )

    def _learn(self, update_meta: bool = False) -> dict[str, float]:
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
            weights = jnp.squeeze(
                self.convert_variable_into_batch(
                    self._kwargs["w_critic"], batch_size=self._kwargs["batch_size"]
                )
            )

        if self._kwargs["use_double_q"]:
            loss_fn = self.double_q_loss_fn
        else:
            loss_fn = self.loss_fn

        reward_from_env = transitions.r_t

        metrics = dict()

        logging.log_first_n(logging.INFO, "Begin learning Critic", 1)

        # update critic
        (
            self._train_rng_key,
            self._opt_state,
            self._online_params,
            target_Q,
            q_1,
            sf_1,
            critic_loss,
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
            optimizer_critic_sf=self._optimizer,
            grad_error_bound=self._kwargs["grad_error_bound"],
            batch_size=self._kwargs["batch_size"],
            weights=weights,
            log_grads=self._kwargs["log_grads"],
            td_error_scaler_sigma=self.td_sigma,
            ewc_state=self._ewc_state,
            ewc_lambda=self._ewc_lambda,
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

        if update_meta:
            logging.log_first_n(logging.INFO, "Begin learning Meta", 1)
            metric_meta = self._update_meta(
                rng_key=self._train_rng_key,
                transitions=transitions,
                optimizer_task=self._optimizer_task,
                opt_state_task=self._opt_state_task,
                batch_size=self._kwargs["batch_size"],
                network=self._network,
                online_params=self._online_params,
                task_params=self._task_params,
                loss_reward_prediction_fn=self.loss_reward_prediction_fn,
                convert_variable_into_batch=self.convert_variable_into_batch,
                regress_meta_using_grad_descent=self._regress_meta_using_grad_descent,
                weights=weights,
                log_grads=self._kwargs["log_grads"],
                normalize_task_params=self._kwargs["normalize_task_params"],
            )
            self._solved_meta = metric_meta["meta"]
            self._opt_state_task = metric_meta["opt_state_task"]
            self._task_params = metric_meta["task_params"]
            self._train_rng_key = metric_meta["rng_key"]

            metric_meta = self.del_unwanted_items_from_metric(metric_meta)

            for key, value in metric_meta.items():
                if key != "meta":
                    metric_meta[key] = value.item()

            metrics.update(metric_meta)

        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["critic_q1"] = q_1.mean().item()
        metrics["sf_1"] = sf_1.mean().item()
        metrics["critic_loss"] = critic_loss.item()
        metrics["extr_reward"] = reward_from_env.mean().item()
        metrics["exploration_epsilon"] = self.exploration_epsilon
        metrics["td_error_scaler_sigma"] = self.td_sigma.item()

        # average norm of sf per minibatch sample
        metrics["sf_norm"] = jnp.mean(jnp.linalg.norm(sf_1, ord=2, axis=1)).item()

        if self._kwargs["consolidation"] and (
            self._frame_t % self._fisher_update_interval == 0
        ):
            fisher = compute_fisher_diagonal(
                lambda p: loss_fn(
                    p,
                    self._target_params,
                    transitions,
                    self._train_rng_key,
                    self._network,
                    self._kwargs["grad_error_bound"],
                    self._kwargs["batch_size"],
                    weights,
                    self.td_sigma,
                    self._ewc_state,
                    self._ewc_lambda,
                )[0],
                self._online_params,
                transitions,
            )

            if self._ewc_state is None:
                self._ewc_state = EWCState(
                    params_star=consolidate_params(self._online_params), fisher=fisher
                )
            else:
                updated_fisher = update_fisher(
                    self._ewc_state.fisher, fisher, self._ewc_gamma
                )
                self._ewc_state = EWCState(
                    params_star=consolidate_params(self._online_params),
                    fisher=updated_fisher,
                )

        return metrics
