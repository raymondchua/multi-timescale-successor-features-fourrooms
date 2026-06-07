from typing import Any, Callable, Mapping, Text, Tuple, Sequence, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import chex
import dm_env
import wandb
import distrax
import sys
from flax.core import freeze, unfreeze
import jax.tree_util as tree_util


from functools import partial

from .sf_consolidation_params_continuous_agent import (
    SFConsolidationParamsContinuousAgent,
    SFConsolidationParamsContinuousAgentKwargs,
    Transition_Task,
    Params,
    Array,
    PRNGKey,
    Network,
)

TaskParams = Mapping[str, Any]

from .simple_sf_agent import Action

from typing_extensions import Unpack
from network import (
    SFSoftmaxAttentionDiffUniqueNetwork,
)

from helpers import optimizers
from helpers.successor_features import get_all_sf
from losses import update_and_accumulate_tree, pytree_l2_norm

from collections import OrderedDict

from absl import logging

# Batch variant of functions for learning.
_batch_q_learning = jax.vmap(rlax.q_learning)
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)


class SFConsolidationParamsContinuousSoftmaxAttentionDiffUniqueAgentKwargs(
    SFConsolidationParamsContinuousAgentKwargs
):
    replace_zero_logits: bool
    optimizer_attention: str
    apply_gain_sf_diff: bool
    apply_mask_to_keys: bool
    layer_norm_keys: bool
    layer_norm_values: bool
    learnable_layer_norm_parameters: bool


class SFConsolidationParamsContinuousSoftmaxAttentionDiffUniqueAgent(
    SFConsolidationParamsContinuousAgent
):
    def __init__(
        self,
        **kwargs: Unpack[SFConsolidationParamsContinuousSoftmaxAttentionDiffUniqueAgentKwargs],
    ):

        super().__init__(**kwargs)

        self._attention_network = self.get_attention_network_fn()
        self._sample_sf_input = jnp.zeros(
            (
                1,
                self._kwargs["num_beakers"],
                self._kwargs["action_shape"],
                self._kwargs["sf_dim"],
            )
        )

        self._sample_mask_input = jnp.zeros((self._kwargs["num_beakers"],))
        self._sample_recall_gain_input = jnp.zeros((self._kwargs["num_beakers"] - 1,))

        self._online_attention_params = self._attention_network.init(
            self._network_rng_key,
            self._sample_sf_input,
            self._sample_task_input_extended,
            self._sample_mask_input,
            self._sample_recall_gain_input,
        )["params"]

        self._target_attention_params = self._online_attention_params
        self._optimizer_attention = optimizers.get_optimizer(
            self._kwargs["optimizer_attention"],
            self._kwargs["lr"],
        )
        self._opt_state_attention = self._optimizer_attention.init(
            self._online_attention_params
        )
        self._mask = jnp.ones((self._kwargs["num_beakers"],), dtype=jnp.int32)
        self._recall_gain = None


    @staticmethod
    def loss_attention_fn(
        online_params: Params,
        target_params: Params,
        online_attention_params: Params,
        target_attention_params: Params,
        transitions: Transition_Task,
        network: Network,
        attention_network: Network,
        grad_error_bound: float,
        batch_size: int,
        weights: Array,
        td_error_scaler_sigma: Array,
        online_sf_consolidation: Array,
        mask: Array,
        recall_gain: Array,
    ) -> [float, (Array, Array, Array, Array, Array, Array)]:
        """

        Parameters
        ----------
        online_params: Params
        target_params: Params
        online_attention_params: Params
        target_attention_params: Params
        transitions: Transition
        network: Network
        attention_network: Network
        grad_error_bound: float
        batch_size: int
        weights: Array - Importance sampling weights for prioritized replay
        td_error_scaler_sigma: Array
        online_sf_consolidation: Array
        mask: Array - Mask to mask out the beakers in the consolidation system which has timescales less than the current time
        recall_gain: Array - Recall gain for each beaker

        Returns
        -------
        critic_loss: Array
        td_errors: Array - TD errors for prioritized replay

        """
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

        target_sf = jnp.expand_dims(jnp.swapaxes(target_output.sf, 1, 2), 1)
        online_sf = jnp.expand_dims(jnp.swapaxes(online_output.sf, 1, 2), 1)

        online_sf_consolidation = jnp.swapaxes(online_sf_consolidation, 2, 3)

        online_sf_consolidation = jax.lax.stop_gradient(online_sf_consolidation)

        # Concatenate each tensor with online_sf_consolidation along the second axis
        target_sf = jnp.concatenate([target_sf, online_sf_consolidation], axis=1)
        online_sf = jnp.concatenate([online_sf, online_sf_consolidation], axis=1)

        online_attention_output = attention_network.apply(
            {"params": online_attention_params},
            online_sf,
            transitions.task,
            mask,
            recall_gain,
        )

        target_attention_output = attention_network.apply(
            {"params": target_attention_params},
            target_sf,
            transitions.task,
            mask,
            recall_gain,
        )

        target_Q1 = target_attention_output.q_1
        q_1 = online_attention_output.q_1
        attention_logits = online_attention_output.attention_logits
        attention_outputs = online_attention_output.attention_outputs
        attended_sf = online_attention_output.attended_sf

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
        return loss, (
            td_errors,
            attention_logits,
            attention_outputs,
            target_Q1,
            q_1,
            attended_sf,
        )

    @staticmethod
    def double_q_loss_attention_fn(
        online_params: Params,
        target_params: Params,
        online_attention_params: Params,
        target_attention_params: Params,
        transitions: Transition_Task,
        network: Network,
        attention_network: Network,
        grad_error_bound: float,
        batch_size: int,
        weights: Array,
        td_error_scaler_sigma: Array,
        online_sf_consolidation: Array,
        mask: Array,
        recall_gain: Array,
    ) -> [float, (Array, Array, Array, Array, Array, Array)]:
        """

        Parameters
        ----------
        online_params: Params
        target_params: Params
        online_attention_params: Params
        target_attention_params: Params
        transitions: Transition
        network: Network
        attention_network: Network
        grad_error_bound: float
        batch_size: int
        weights: Array - Importance sampling weights for prioritized replay
        td_error_scaler_sigma: Array
        online_sf_consolidation: Array
        mask: Array - Mask to mask out the beakers in the consolidation system which has timescales less than the current time
        recall_gain: Array - Multiply the SF values based on the recall g_flow / capacity if using pre-defined gains

        Returns
        -------
        critic_loss: Array
        td_errors: Array - TD errors for prioritized replay

        """

        online_tm1_output = network.apply(
            {"params": online_params},
            transitions.s_tm1,
            transitions.task,
        )

        online_t_output = network.apply(
            {"params": online_params},
            transitions.s_t,
            transitions.task,
        )

        target_output = network.apply(
            {"params": target_params},
            transitions.s_t,
            transitions.task,
        )

        target_sf = jnp.expand_dims(jnp.swapaxes(target_output.sf, 1, 2), 1)
        online_tm1_sf = jnp.expand_dims(jnp.swapaxes(online_tm1_output.sf, 1, 2), 1)
        online_t_sf = jnp.expand_dims(jnp.swapaxes(online_t_output.sf, 1, 2), 1)

        online_sf_consolidation = jnp.swapaxes(online_sf_consolidation, 2, 3)

        online_sf_consolidation = jax.lax.stop_gradient(online_sf_consolidation)

        # Concatenate each tensor with online_sf_consolidation along the second axis
        target_sf = jnp.concatenate([target_sf, online_sf_consolidation], axis=1)
        online_tm1_sf = jnp.concatenate(
            [online_tm1_sf, online_sf_consolidation], axis=1
        )
        online_t_sf = jnp.concatenate([online_t_sf, online_sf_consolidation], axis=1)

        online_attention_tm1_output = attention_network.apply(
            {"params": online_attention_params},
            online_tm1_sf,
            transitions.task,
            mask,
            recall_gain,
        )

        online_attention_t_output = attention_network.apply(
            {"params": online_attention_params},
            online_t_sf,
            transitions.task,
            mask,
            recall_gain,
        )

        target_attention_output = attention_network.apply(
            {"params": target_attention_params},
            target_sf,
            transitions.task,
            mask,
            recall_gain,
        )

        q_target_t = target_attention_output.q_1
        q_tm1 = online_attention_tm1_output.q_1
        q_t = online_attention_t_output.q_1

        attention_logits = online_attention_tm1_output.attention_logits
        attention_outputs = online_attention_tm1_output.attention_outputs
        attended_sf = online_attention_tm1_output.attended_sf

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
        return loss, (
            td_errors,
            attention_logits,
            attention_outputs,
            q_target_t,
            q_tm1,
            attended_sf,
        )

    def get_attention_network_fn(self) -> Network:
        return SFSoftmaxAttentionDiffUniqueNetwork(
            num_actions=self._kwargs["action_shape"],
            sf_dim=self._kwargs["sf_dim"],
            num_beakers=self._kwargs["num_beakers"],
            apply_mask_to_keys=self._kwargs["apply_mask_to_keys"],
            apply_gain_sf_diff=self._kwargs["apply_gain_sf_diff"],
            layer_norm_keys=self._kwargs["layer_norm_keys"],
            layer_norm_values=self._kwargs["layer_norm_values"],
            learnable_layer_norm_parameters=self._kwargs["learnable_layer_norm_parameters"],
        )

    def step(
        self,
        timestep: dm_env.TimeStep,
        time_to_learn: bool = False,
        learn_meta: bool = False,
        task: Array = None,
        task_id: int = 0,
        action: int = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Performs a step in the environment and store the transition in replay.
        Parameters
        ----------
        timestep
        time_to_learn
        learn_meta
        task
        task_id : int - the task id of the current task.
            If it is the first task, then we do not use the back flow consolidation loss.
        action : int - If action is provided, such as from a pre-trained policy, we use that action instead of sampling.
        kwargs

        Returns
        -------

        """
        self._frame_t += 1
        self._frame_t_exploration += 1
        self._update_step = int(self._frame_t)
        metrics = dict()
        timestep = self._preprocessor(timestep)
        task_no_grad = jax.lax.stop_gradient(task)
        update_consolidation = self._frame_t % self._update_consolidation_every_steps == 0

        """
        Make a mask to mask out the beakers in the consolidation system which has timescales less than the current time
        step. 
        """
        self._mask = self.compute_recall_mask(self._update_step, self._recall_timescales)

        if self._recall_gain is None:
            g_flow_array = jnp.ones(self._kwargs["num_beakers"] - 1) * self._g_flow
            self._recall_gain = g_flow_array / self._capacity[1:-1]

        self._train_rng_key, act_rng_key = jax.random.split(self._train_rng_key, 2)

        if timestep is None:  # Repeat action.
            if action is None:
                action = self._action

            else:
                self._action = action

        else:
            if action is None:
                # action = self._action = self._act(timestep, task_no_grad, mask)
                train_rng_key, action, self._statistics = self._act_static(
                    act_rng_key,
                    timestep,
                    task_no_grad,
                    self._mask,
                    self._network_params,
                    self._online_attention_params,
                    self.exploration_epsilon,
                    self._network,
                    self._attention_network,
                    self._statistics,
                    self._num_beakers,
                    self._select_action,
                    self._recall_gain,
                )
                self._action = action
            else:
                self._action = action
                # self._act(
                #     timestep, task_no_grad, mask
                # )  # just to store the state value, action is not being used
                self._act_static(
                    act_rng_key,
                    timestep,
                    task_no_grad,
                    self._mask,
                    self._network_params,
                    self._online_attention_params,
                    self.exploration_epsilon,
                    self._network,
                    self._attention_network,
                    self._statistics,
                    self._num_beakers,
                    self._select_action,
                    self._recall_gain,
                )

            for transition in self._transition_accumulator.step(
                timestep, action, task_no_grad
            ):
                if self._kwargs["use_priority_replay"]:
                    self._replay.add(transition, priority=self._max_seen_priority)

                else:
                    self._replay.add(transition)

        if time_to_learn and learn_meta:
            learn_metrics = self._learn(
                mask=self._mask,
                update_meta=True,
                update_consolidation=update_consolidation,
            )

            # best to log the wandb metric here for the training loss
            metrics.update(learn_metrics)

        # update meta parameters only if the update meta flag is set
        elif learn_meta and not time_to_learn:
            transitions_meta = self._replay.uniform_sample(self._kwargs["batch_size"])
            weights = jnp.ones(self._kwargs["batch_size"])
            metric_meta = self._update_meta(
                rng_key=self._train_rng_key,
                transitions=transitions_meta,
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
                    metric_meta[key] = value

            metrics.update(metric_meta)

        elif time_to_learn and not learn_meta:
            learn_metrics = self._learn(
                mask=self._mask,
                update_meta=False,
                update_consolidation=update_consolidation,
            )

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
                self._target_attention_params = jax.tree_util.tree_map(
                    lambda x, y: self._kwargs["critic_target_tau"] * x
                    + ((1 - self._kwargs["critic_target_tau"]) * y),
                    self._online_attention_params,
                    self._target_attention_params,
                )
            else:
                self._target_params = self._online_params
                self._target_attention_params = self._online_attention_params

        metrics["action"] = action
        metrics["exploration_epsilon"] = self.exploration_epsilon

        if self._kwargs["print_metrics"]:
            for key, value in metrics.items():
                if key != "meta":
                    print(f"{key}: {value}")

        return metrics

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "network",
            "attention_network",
            "loss_fn",
            "optimizer_sf",
            "optimizer_attention",
            "grad_error_bound",
            "batch_size",
            "log_grads",
        ],
    )
    def _update(
        rng_key: PRNGKey,
        opt_state: optax.OptState,
        opt_state_attention: optax.OptState,
        online_params: Params,
        target_params: Params,
        online_attention_params: Params,
        target_attention_params: Params,
        transitions: Transition_Task,
        network: Network,
        attention_network: Network,
        loss_fn: Callable,
        optimizer_sf: optax.GradientTransformation,
        optimizer_attention: optax.GradientTransformation,
        td_error_scaler_sigma: Array,
        grad_error_bound: float,
        batch_size: int,
        weights: Array,
        sf_consolidation: Array,
        mask: Array,
        recall_gain: Array,
        log_grads: bool = False,
    ):
        rng_key, apply_key, target_key, loss_key, policy_key = jax.random.split(
            rng_key, 5
        )

        (
            critic_loss,
            (
                td_errors,
                attention_logits,
                attention_outputs,
                target_Q,
                q_1,
                attended_sf,
            ),
        ), (grads_online, grads_attention,) = jax.value_and_grad(
            loss_fn, has_aux=True, argnums=(0, 2)
        )(
            online_params,
            target_params,
            online_attention_params,
            target_attention_params,
            transitions,
            network,
            attention_network,
            grad_error_bound,
            batch_size,
            weights,
            td_error_scaler_sigma,
            sf_consolidation,
            mask,
            recall_gain,
        )

        updates, new_opt_state = optimizer_sf.update(grads_online, opt_state)
        new_online_params = optax.apply_updates(online_params, updates)

        updates_attention, new_opt_state_attention = optimizer_attention.update(
            grads_attention, opt_state_attention
        )
        new_online_attention_params = optax.apply_updates(
            online_attention_params, updates_attention
        )

        if log_grads:
            grads = grads_online
            grads_attention = grads_attention
        else:
            grads = None
            grads_attention = None

        return (
            rng_key,
            new_opt_state,
            new_opt_state_attention,
            new_online_params,
            new_online_attention_params,
            target_Q,
            q_1,
            critic_loss,
            td_errors,
            grads,
            grads_attention,
            attention_logits,
            attention_outputs,
            attended_sf,
        )

    def _learn(self, mask: Array, update_meta: bool = False, update_consolidation: bool = True) -> dict[str, float]:
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
            loss_fn = self.double_q_loss_attention_fn
        else:
            loss_fn = self.loss_attention_fn

        reward_from_env = transitions.r_t

        metrics = dict()

        network = self._network
        online_sf_all = get_all_sf(
            transitions=transitions,
            num_beakers=self._num_beakers,
            network_params=self._network_params,
            network=self._network,
        )
        online_sf_consolidation = online_sf_all[:, 1:, :]  # ignore the first beaker

        # update critic
        (
            self._train_rng_key,
            self._opt_state,
            self._opt_state_attention,
            self._online_params,
            self._online_attention_params,
            target_Q,
            q_1,
            critic_loss,
            td_errors,
            grads,
            grads_attention,
            attention_logits,
            attention_outputs,
            attended_sf,
        ) = self._update(
            rng_key=self._train_rng_key,
            opt_state=self._opt_state,
            opt_state_attention=self._opt_state_attention,
            online_params=self._online_params,
            target_params=self._target_params,
            online_attention_params=self._online_attention_params,
            target_attention_params=self._target_attention_params,
            transitions=transitions,
            network=network,
            attention_network=self._attention_network,
            loss_fn=loss_fn,
            optimizer_sf=self._optimizer,
            optimizer_attention=self._optimizer_attention,
            grad_error_bound=self._kwargs["grad_error_bound"],
            batch_size=self._kwargs["batch_size"],
            weights=weights,
            log_grads=self._kwargs["log_grads"],
            td_error_scaler_sigma=self.td_sigma,
            sf_consolidation=online_sf_consolidation,
            mask=mask,
            recall_gain=self._recall_gain,
        )

        self._network_params[0] = self._online_params

        if update_consolidation:
            # update consolidation
            (
                self._network_params,
                consolidation_loss,
                network_params_norm,
            ) = self.consolidation_update_fn(
                params=self._network_params,
                params_set_to_zero=self._params_set_to_zero,
                g_flow=self._g_flow,
                capacity=self._capacity,
                mask=mask,
                num_beakers=self._kwargs["num_beakers"],
                lr_consolidation=self._lr_consolidation,
                update_consolidation_every_steps=self._update_consolidation_every_steps,
            )

            metrics["consolidation_loss"] = consolidation_loss

            for i in range(self._kwargs["num_beakers"]):
                metrics[f"params_u{i}_norm"] = network_params_norm[i]

        if self._kwargs["log_grads"]:
            assert grads is not None
            metrics = self.log_grads_norm(
                metrics=metrics, gradients=grads, name="critic"
            )
            metrics = self.log_grads_norm(
                metrics=metrics, gradients=grads_attention, name="attention"
            )
            values_gain_grad = grads_attention["sf_values_gain"]
            metrics[f"values_gain_grad_norm"] = jnp.linalg.norm(
                values_gain_grad
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
                    metric_meta[key] = value

            metrics.update(metric_meta)

        metrics["critic_target_q"] = target_Q.mean()
        metrics["critic_q1"] = q_1.mean()
        metrics["critic_loss"] = critic_loss
        metrics["extr_reward"] = reward_from_env.mean()
        metrics["exploration_epsilon"] = self.exploration_epsilon
        metrics["td_error_scaler_sigma"] = self.td_sigma

        attention_outputs = jnp.squeeze(attention_outputs)
        attention_logits = jnp.squeeze(attention_logits)

        online_sf_all = jnp.swapaxes(online_sf_all, 2, 3)
        chex.assert_shape(online_sf_all, (
            self._kwargs["batch_size"],
            self._kwargs["num_beakers"],
            self._kwargs["action_shape"],
            self._kwargs["sf_dim"],
        ))

        chex.assert_shape(attention_outputs, (
            self._kwargs["batch_size"],
            self._kwargs["num_beakers"] - 1,
            self._kwargs["action_shape"],
        ))

        # loop over beakers and store the sf for each beaker and action
        for i in range(attention_outputs.shape[1]):
            # we compute the mean across the action dimension to when logging the attention outputs
            attention_outputs_current_beaker = attention_outputs[:, i, :]
            metrics[
                f"attention_outputs_u{i}_mean"
            ] = attention_outputs_current_beaker.mean()
            metrics[
                f"attention_outputs_u{i}_std"
            ] = attention_outputs_current_beaker.std()

            attention_logits_current_beaker = attention_logits[:, i, :]
            metrics[
                f"attention_logits_u{i}_mean"
            ] = attention_logits_current_beaker.mean()
            metrics[
                f"attention_logits_u{i}_std"
            ] = attention_logits_current_beaker.std()

            metrics[f"mask_u{i}"] = mask[i]

        if self._kwargs["apply_gain_sf_diff"]:
            metrics[f"values_gain_norm"] = jnp.linalg.norm(
                self._online_attention_params["sf_values_gain"]
            )

        return metrics

    @staticmethod
    @partial(
        jax.jit, static_argnames=("attention_network", "num_beakers", "sf_network", "select_action_fn")
    )
    def _act_static(
        train_rng_key: PRNGKey,
        timestep: dm_env.TimeStep,
        task: Array,
        mask: Array,
        network_params: Params,
        online_attention_params: Params,
        exploration_epsilon: float,
        sf_network: Network,
        attention_network: Network,
        statistics: dict,
        num_beakers: int,
        select_action_fn: Callable,
        recall_gain: Array,
    ) -> Tuple[PRNGKey, Action, dict]:
        """Selects action given timestep, according to epsilon-greedy policy."""
        task_no_grad = jax.lax.stop_gradient(task)
        task_no_grad = (
            jnp.expand_dims(task_no_grad, axis=0)
            if task_no_grad.ndim == 1
            else task_no_grad
        )

        # Cast as transition to get SFs from all the networks
        transition = Transition_Task(
            s_tm1=None,
            a_tm1=None,
            r_t=None,
            discount_t=None,
            s_t=jnp.expand_dims(timestep.observation, axis=0)
            if timestep.observation.ndim == 3
            else timestep.observation,
            task=task_no_grad,
        )

        online_sf_all = get_all_sf(
            transitions=transition,
            num_beakers=num_beakers,
            network_params=network_params,
            network=sf_network,
        )

        # (Batch, num_beakers, action_shape, sf_dim)
        online_sf_all = jnp.swapaxes(online_sf_all, 2, 3)

        train_rng_key, a_t, v_t = select_action_fn(
            train_rng_key,
            online_attention_params,
            online_sf_all,
            exploration_epsilon,
            task_no_grad,
            attention_network,
            mask,
            recall_gain,
        )

        a_t, v_t = jax.device_get((a_t, v_t))
        a_t = jnp.squeeze(a_t)
        statistics["state_value"] = jnp.squeeze(v_t)
        return train_rng_key, a_t, statistics

    def act_eval(
        self,
        timestep: dm_env.TimeStep,
        task: Array,
        mask: Array,
        **kwargs: Any,
    ) -> Mapping[Text, Any]:
        """Selects action given timestep and with no learning involved."""
        timestep = self._preprocessor(timestep)
        metric = dict()

        if timestep is None:  # Repeat action.
            metric["action"] = self._action
            return metric

        task_no_grad = jax.lax.stop_gradient(task)
        task_no_grad = (
            jnp.expand_dims(task_no_grad, axis=0)
            if task_no_grad.ndim == 1
            else task_no_grad
        )

        # cast as transition to get SFs from all the networks
        transition = Transition_Task(
            s_tm1=None,
            a_tm1=None,
            r_t=None,
            discount_t=None,
            s_t=jnp.expand_dims(timestep.observation, axis=0)
            if timestep.observation.ndim == 3
            else timestep.observation,
            task=task_no_grad,
        )
        online_sf_all = get_all_sf(
            transitions=transition,
            num_beakers=self._num_beakers,
            network_params=self._network_params,
            network=self._network,
        )

        # (Batch, num_beakers, action_shape, sf_dim)
        online_sf_all = jnp.swapaxes(online_sf_all, 2, 3)

        self._rng_key, a_t, v_t = self._select_action(
            self._rng_key,
            self._online_attention_params,
            online_sf_all,
            self.eval_exploration_epsilon,
            task_no_grad,
            self._attention_network,
            mask,
            self._recall_gain,
        )
        self._rng_key, pred_reward = self._get_predicted_reward(
            rng_key=self._rng_key,
            online_params=self._online_params,
            s_t=timestep.observation,
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
    @partial(jax.jit, static_argnames="attention_network")
    def _select_action(
        rng_key: PRNGKey,
        online_params: Params,
        sf_all: Array,
        exploration_epsilon: float,
        task: Array,
        attention_network: Network,
        mask: Array,
        recall_gain: Array,
    ) -> tuple[PRNGKey, Array, Array]:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        rng_key, sf_critic_key, policy_key = jax.random.split(rng_key, 3)
        q_t = attention_network.apply(
            {"params": online_params},
            sf_all,
            task,
            mask,
            recall_gain,
        ).q_1
        a_t = distrax.EpsilonGreedy(
            preferences=q_t, epsilon=exploration_epsilon
        ).sample(seed=policy_key)
        v_t = jnp.max(q_t, axis=-1)
        return rng_key, a_t, v_t

    @property
    def network_params(self) -> list[Params]:
        return self._network_params

    @property
    def attention_network(self) -> Network:
        return self._attention_network

    @property
    def attention_network_params(self) -> Params:
        return self._online_attention_params

    @property
    def mask(self) -> Array:
        return self._mask

    @staticmethod
    @partial(jax.jit, static_argnames=["num_beakers", "lr_consolidation", "update_consolidation_every_steps"])
    def consolidation_update_fn(
        params: List[Params],
        params_set_to_zero: Params,
        capacity: Array,
        g_flow: Array,
        mask: Array,
        num_beakers: int,
        lr_consolidation: float,
        update_consolidation_every_steps: int,
    ) -> Tuple[List[Params], float, Array]:
        """
        Consolidation update function for the beakers. Difference is that there
        is no recall for the first beaker as it is handled by the attention network.
        :param params:
        :param params_set_to_zero:
        :param g_flow:
        :param capacity:
        :param mask:
        :param num_beakers:
        :param lr_consolidation:
        :param update_consolidation_every_steps:
        :return:
        """
        loss = 0.0
        params_norm = jnp.zeros(num_beakers)

        # Stack list of PyTrees into a PyTree of arrays
        params_stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *params)

        def get_beaker(ps, i):
            return jax.tree_util.tree_map(lambda x: x[i], ps)

        def set_beaker(ps, i, new_p):
            return jax.tree_util.tree_map(lambda x, new_x: jax.lax.dynamic_update_index_in_dim(x, new_x, i, axis=0), ps,
                                          new_p)
        def scan_body_fn(carry, i):
            ps, loss = carry
            p_prev = get_beaker(ps, i - 1)
            p_i = get_beaker(ps, i)

            scale_prev = g_flow / capacity[i]

            # Consolidate from previous
            p_i, loss = update_and_accumulate_tree(p_i, p_prev, scale_prev * update_consolidation_every_steps, loss)

            ps = set_beaker(ps, i, p_i)
            norm = pytree_l2_norm(p_i)
            return (ps, loss), norm

        # Scan over middle beakers
        (params_stacked, loss), norms = jax.lax.scan(
            scan_body_fn,
            (params_stacked, loss),
            jnp.arange(1, num_beakers - 1)
        )
        params_norm = params_norm.at[1:num_beakers - 1].set(norms)

        # Last beaker
        p_last = get_beaker(params_stacked, num_beakers - 1)
        p_second_last = get_beaker(params_stacked, num_beakers - 2)
        scale_last = g_flow / capacity[-1]
        scale_second_last = g_flow / capacity[-1]

        p_last, loss = update_and_accumulate_tree(p_last, params_set_to_zero, scale_last, loss)
        p_last, loss = update_and_accumulate_tree(
            p_last, p_second_last,
            scale_second_last * lr_consolidation * update_consolidation_every_steps,
            loss
        )
        params_stacked = set_beaker(params_stacked, num_beakers - 1, p_last)
        params_norm = params_norm.at[num_beakers - 1].set(pytree_l2_norm(p_last))

        # Unstack back into list of PyTrees
        final_params = [jax.tree_util.tree_map(lambda x: x[i], params_stacked) for i in range(num_beakers)]
        return final_params, loss, params_norm

    @property
    def recall_gain(self) -> Array:
        if self._recall_gain is not None:
            return self._recall_gain
        else:
            return jnp.zeros(self._kwargs["num_beakers"] - 1)

