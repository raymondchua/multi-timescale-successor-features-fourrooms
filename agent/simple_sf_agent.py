import jax
import rlax
from typing import Unpack, Callable, Tuple, Any, Text, Mapping, Union
import chex
import distrax
import optax
import dm_env
import pickle

from dm_env import specs

from .BaseAgent import (
    Array,
    BaseAgent,
    BaseAgentKwargs,
    Network,
    Params,
    PRNGKey,
    Transition,
    Action,
)
from network import SFNetworkMinatar, SFNetOutputs, SFNetwork
from jax import random

import jax.numpy as jnp
import numpy as np

from functools import partial
from helpers import (
    optimizers,
    processors,
    NStepTransitionTaskAccumulator,
    ReplayStructure,
    TransitionTaskAccumulator,
    Transition_Task,
)

from losses import reward_prediction_loss

from absl import logging
from pathlib import Path
from collections import OrderedDict

TaskParams = Mapping[str, Any]

# Batch variant of functions for learning.
_batch_reward_prediction_loss = jax.vmap(reward_prediction_loss)
_batch_q_learning = jax.vmap(rlax.q_learning)
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)


class SimpleSFAgentKwargs(BaseAgentKwargs):
    feature_dim: int
    hidden_dim: int
    lr_task: float
    normalize_task_params: bool
    optimizer_task: str
    reward_free: bool
    sf_dim: int
    task_dependent_sf: bool
    task_tau: float
    update_basis_features_from_sf: bool
    update_task_every_step: int
    use_framestack: bool
    use_soft_task_update: bool
    use_plasticity_injection: bool
    w_critic: float


class SimpleSFAgent(BaseAgent):
    def __init__(
        self,
        **kwargs: Unpack[SimpleSFAgentKwargs],
    ):
        super().__init__(**kwargs)

        self._optimizer_task = optimizers.get_optimizer(
            self._kwargs["optimizer_task"], self._kwargs["lr_task"]
        )

        self._solved_meta = None
        self._task_params = {"w": self.init_meta()}

        # Initialize critic SF network parameters and optimizer.
        self._sample_task_input = jnp.zeros_like(self._task_params["w"])
        self._sample_task_input_extended = self._sample_task_input[None, ...]

        self._network = self.get_network_fn()
        self._network_eval = self._network

        # initialise critic sf network and params
        self._online_params = self._network.init(
            self._network_rng_key,
            self._sample_network_input_extended,
            self._sample_task_input_extended,
        )["params"]

        self._target_params = self._online_params
        self._opt_state = self._optimizer.init(self._online_params)
        self._opt_state_task = self._optimizer_task.init(self._task_params)

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
        return loss, td_errors

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
        return loss, td_errors

    @staticmethod
    def loss_reward_prediction_fn(
        task_params: Params,
        transitions: Transition_Task,
        basis_features: Array,
        batch_size: int,
        weights: Array,
    ) -> Array:
        """Calculates loss given network parameters and transitions."""
        task_batch = SimpleSFAgent.convert_variable_into_batch(
            task_params["w"], batch_size
        )
        reward_prediction_losses = _batch_reward_prediction_loss(
            basis_features, task_batch, transitions.r_t
        )
        chex.assert_shape(
            reward_prediction_losses, (batch_size,)
        )  # just to check the shape of the losses matches the batch size
        return jnp.mean(reward_prediction_losses * weights)

    @staticmethod
    @partial(jax.jit, static_argnames="network")
    def _get_output(
        rng_key: PRNGKey,
        online_params: Params,
        s_t: Array,
        task: Array,
        network: Network,
    ) -> SFNetOutputs:

        output = network.apply(
            {"params": online_params},
            s_t[None, ...],
            task[None, ...],
        )
        return jax.lax.stop_gradient(output)

    @staticmethod
    @partial(jax.jit, static_argnames="network")
    def _get_predicted_reward(
        rng_key: PRNGKey,
        online_params: Params,
        s_t: Array,
        task: Array,
        network: Network,
    ) -> tuple[PRNGKey, Array]:

        basis_features_t = network.apply(
            {"params": online_params},
            s_t[None, ...],
            task[None, ...],
        ).basis_features

        pred_reward = jnp.dot(jnp.squeeze(basis_features_t), jnp.squeeze(task))
        return rng_key, jax.lax.stop_gradient(pred_reward)

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "batch_size",
            "loss_reward_prediction_fn",
            "optimizer_task",
            "log_grads",
        ],
    )
    def _regress_meta_using_grad_descent(
        rng_key: PRNGKey,
        opt_state_task: optax.OptState,
        task_params: Params,
        transitions: Transition_Task,
        basis_features_t: Array,
        batch_size: int,
        loss_reward_prediction_fn: Callable,
        weights: Array,
        optimizer_task: optax.GradientTransformation,
        log_grads: bool = False,
    ) -> dict[str, Any]:

        output = dict()

        loss, d_loss_d_params = jax.value_and_grad(loss_reward_prediction_fn)(
            task_params, transitions, basis_features_t, batch_size, weights
        )

        updates, new_opt_state_task = optimizer_task.update(
            d_loss_d_params, opt_state_task
        )

        # get new learning rate from new_opt_state_task
        new_lr_task = new_opt_state_task.hyperparams["learning_rate"]
        new_task_params = optax.apply_updates(task_params, updates)

        output["rng_key"] = rng_key
        output["new_opt_state_task"] = new_opt_state_task
        output["new_task_params"] = new_task_params
        output["loss"] = loss
        output["new_lr_task"] = new_lr_task

        # log gradients if flag is set
        if log_grads:
            grads_norm = jax.tree_util.tree_map(jnp.linalg.norm, d_loss_d_params)
            output["grads_norm_w"] = jax.device_get(grads_norm["w"])

            # compute the magnitude of the updates
            updates_norm = jax.tree_util.tree_map(jnp.linalg.norm, updates)
            output["updates_norm_w"] = jax.device_get(updates_norm["w"])

        return output

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

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "network",
            "batch_size",
            "loss_reward_prediction_fn",
            "convert_variable_into_batch",
            "regress_meta_using_grad_descent",
            "optimizer_task",
            "log_grads",
            "normalize_task_params",
        ],
    )
    def _update_meta(
        rng_key: PRNGKey,
        transitions: Transition_Task,
        optimizer_task: optax.GradientTransformation,
        opt_state_task: optax.OptState,
        network: Network,
        online_params: Params,
        task_params: TaskParams,
        loss_reward_prediction_fn: Callable,
        convert_variable_into_batch: Callable,
        regress_meta_using_grad_descent: Callable,
        batch_size: int,
        weights: Array,
        log_grads: bool = False,
        normalize_task_params: bool = False,
    ) -> dict[str, Any]:

        metric = dict()
        rng_key, apply_key, loss_key = jax.random.split(rng_key, 3)

        task_batch = convert_variable_into_batch(task_params["w"], batch_size)
        task_batch_no_grad = jax.lax.stop_gradient(task_batch)

        output = network.apply(
            {"params": online_params},
            transitions.s_t,
            transitions.task,
        )

        basis_features_t = jax.lax.stop_gradient(jnp.squeeze(output.basis_features))

        output = regress_meta_using_grad_descent(
            rng_key=loss_key,
            opt_state_task=opt_state_task,
            task_params=task_params,
            transitions=transitions,
            basis_features_t=basis_features_t,
            batch_size=batch_size,
            loss_reward_prediction_fn=loss_reward_prediction_fn,
            weights=weights,
            optimizer_task=optimizer_task,
            log_grads=log_grads,
        )

        rng_key = output["rng_key"]
        new_opt_state_task = output["new_opt_state_task"]
        new_task_params = output["new_task_params"]
        new_lr_task = output["new_lr_task"]

        # normalize the task params if the normalize task params flag is set
        if normalize_task_params:
            denominator = jnp.clip(
                jnp.linalg.norm(task_params["w"], ord=2, axis=0), a_min=1e-12
            )  # to prevent dividing by zero

            new_task_params["w"] = new_task_params["w"] / denominator

        solved_meta = jax.lax.stop_gradient(jax.device_get(new_task_params["w"]))

        metric["task_params"] = new_task_params
        metric["opt_state_task"] = new_opt_state_task
        metric["rng_key"] = rng_key
        metric["lr_task"] = new_lr_task

        reward_pred_loss = output["loss"]

        # return the average basis features norm for sanity check
        basis_features_norm = jnp.mean(
            jnp.linalg.norm(basis_features_t, ord=2, axis=1, keepdims=True)
        )

        regress_batch_rewards = transitions.r_t
        regress_batch_rewards_mean = jnp.mean(regress_batch_rewards)

        meta_norm = jnp.linalg.norm(solved_meta, ord=2)

        metric["meta"] = solved_meta
        metric["reward_pred_loss"] = reward_pred_loss
        metric["regress_reward_mean"] = regress_batch_rewards_mean
        metric["basis_features_norm"] = basis_features_norm
        metric["meta_norm"] = meta_norm

        # log gradients if flag is set. Only available for gradient descent method.
        if log_grads and "grads_norm_w" in output:
            logging.info("Logging gradients")
            metric["task_grads_norm_w"] = output["grads_norm_w"]
            metric["task_updates_norm_w"] = output["updates_norm_w"]

        return metric

    @staticmethod
    @partial(jax.jit, static_argnames="network")
    def _select_action(
        rng_key: PRNGKey,
        online_params: Params,
        s_t: Array,
        exploration_epsilon: float,
        task: Array,
        network: Network,
    ) -> tuple[PRNGKey, Array, Array]:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        rng_key, sf_critic_key, policy_key = jax.random.split(rng_key, 3)
        q_t = network.apply(
            {"params": online_params},
            s_t[None, ...],
            task[None, ...],
        ).q_1[0]
        a_t = distrax.EpsilonGreedy(
            preferences=q_t, epsilon=exploration_epsilon
        ).sample(seed=policy_key)
        v_t = jnp.max(q_t, axis=-1)
        return rng_key, a_t, v_t

    def get_meta_specs(self) -> specs.Array:
        return specs.Array((self._kwargs["sf_dim"],), np.float32, "task")

    def init_meta(self) -> Array:
        _, task_rng_key = jax.random.split(self._train_rng_key)
        if self._solved_meta is not None:
            return self._solved_meta
        task = random.uniform(task_rng_key, shape=(self._kwargs["sf_dim"],))
        task = task / jnp.linalg.norm(task, ord=2)
        return task

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
        )

        a_t, v_t = jax.device_get((a_t, v_t))
        self._statistics["state_value"] = v_t
        return Action(a_t)

    def step(
        self,
        timestep: dm_env.TimeStep,
        task: Array,
        time_to_learn: bool = False,
        learn_meta: bool = False,
        action: int = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Performs a step in the environment and store the transition in replay.
        Parameters
        ----------
        timestep
        task
        time_to_learn
        learn_meta
        action : int - If action is provided, such as from a pre-trained policy, we use that action instead of sampling.
        kwargs

        Returns
        -------

        """

        assert task is not None, "task should not be None"

        self._frame_t += 1
        self._frame_t_exploration += 1
        metrics = dict()
        timestep = self._preprocessor(timestep)
        task_no_grad = jax.lax.stop_gradient(task)

        if timestep is None:  # Repeat action.
            if action is None:
                action = self._action
            else:
                self._action = action
        else:
            if action is None:
                action = self._action = self._act(timestep, task_no_grad)

            else:
                self._action = action
                self._act(
                    timestep, task_no_grad
                )  # just to store the state value, action is not being used

            for transition in self._transition_accumulator.step(
                timestep, action, task_no_grad
            ):

                if self._kwargs["use_priority_replay"]:
                    self._replay.add(transition, priority=self._max_seen_priority)

                else:
                    self._replay.add(transition)

        if time_to_learn and learn_meta:
            learn_metrics = self._learn(update_meta=True)

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
                    metric_meta[key] = value.item()

            metrics.update(metric_meta)

        elif time_to_learn and not learn_meta:
            learn_metrics = self._learn(update_meta=False)
            metrics.update(learn_metrics)

        # update critic target and task target parameters
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
            td_error_scaler_sigma=self.td_sigma
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

        return metrics

    def save_snapshot(
        self, task_id: int, snapshot_dir: str, step: int, cfg: dict = None
    ) -> None:
        params = jax.device_get(self._online_params)
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_task{task_id}.pkl"
        with snapshot.open("wb") as f:
            pickle.dump(params, f)

        snapshot_task = snapshot_dir / f"task_{task_id}.pkl"
        with snapshot_task.open("wb") as f:
            pickle.dump(self._solved_meta, f)

        if cfg is not None:
            # save the config file as well if it has not been saved before
            cfg_file = snapshot_dir / f"cfg.pkl"

            if cfg_file.exists():
                logging.info("Config file already exists in snapshot directory")

            else:
                with cfg_file.open("wb") as f:
                    pickle.dump(cfg, f)
                logging.info("Config file saved in snapshot directory")

        # log snapshot saved for task and step
        logging.info("Snapshot saved for task %d and step %d", task_id, step)

    def load_snapshot(self, task_id: int, snapshot_dir: str, step: int = None) -> None:
        snapshot_dir = Path(snapshot_dir)

        # check if snapshot_dir exists, if not raise error
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot directory {snapshot_dir} does not exist")

        snapshot = snapshot_dir / f"snapshot_task{task_id}.pkl"

        # check if snapshot exists, if not raise error
        if not snapshot.exists():
            raise ValueError(f"Snapshot {snapshot} does not exist")

        with snapshot.open("rb") as f:
            params = pickle.load(f)
            self._online_params = jax.device_put(params)
            self._target_params = self._online_params
            logging.info("Snapshot loaded for task %d", task_id)

        snapshot_task = snapshot_dir / f"task_{task_id}.pkl"

        # check if snapshot_task exists, if not raise error
        if not snapshot_task.exists():
            raise ValueError(f"Snapshot task {snapshot_task} does not exist")

        with snapshot_task.open("rb") as f:
            meta_loaded = pickle.load(f)
            if meta_loaded is not None:
                self._solved_meta = jax.device_put(meta_loaded)
                self._task_params["w"] = meta_loaded
                logging.info("Snapshot task loaded for task %d", task_id)
            else:
                self._solved_meta = None
                self._task_params = {"w": self.init_meta()}
                logging.info("Snapshot task is None for task %d", task_id)
                logging.info(
                    "Initialising meta parameters for task %d instead", task_id
                )

    def load_task_snapshot(self, task_id: int, snapshot_dir: str) -> None:
        """
        Load the snapshot for the task only. This is useful when we want to load the task snapshot only and not the
        critic.

        Parameters
        ----------
        task_id : int
        snapshot_dir : str

        Returns
        -------
        None

        """
        snapshot_dir = Path(snapshot_dir)

        # check if snapshot_dir exists, if not raise error
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot directory {snapshot_dir} does not exist")

        snapshot_task = snapshot_dir / f"task_{task_id}.pkl"

        # check if snapshot_task exists, if not raise error
        if not snapshot_task.exists():
            raise ValueError(f"Snapshot task {snapshot_task} does not exist")

        with snapshot_task.open("rb") as f:
            meta_loaded = pickle.load(f)
            if meta_loaded is not None:
                self._solved_meta = jax.device_put(meta_loaded)
                self._task_params["w"] = meta_loaded
                logging.info("Snapshot task loaded for task %d", task_id)
            else:
                self._solved_meta = None
                self._task_params = {"w": self.init_meta()}
                logging.info("Snapshot task is None for task %d", task_id)
                logging.info(
                    "Initialising meta parameters for task %d instead", task_id
                )

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

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """
        self._transition_accumulator.reset()
        processors.reset(self._preprocessor)
        self._action = None

    @property
    def update_task_every_step(self) -> int:
        return self._kwargs["update_task_every_step"]

    @property
    def meta(self) -> Array:
        # return self._solved_meta if self._solved_meta is not None else self.init_meta()
        if self._solved_meta is not None:
            return self._solved_meta
        else:
            return jax.device_get(self._task_params["w"])

    @meta.setter
    def meta(self, meta: Array):
        self._solved_meta = meta

    def reset_replay_buffer(self) -> None:
        """Reset the replay buffer of the agent"""
        logging.log(logging.INFO, "Reset Replay buffer")
        self._replay.reset()

    def get_network_fn(self) -> Network:
        """Returns a function that computes the network output. Minatar has a different network since the
        observation is non-pixels and is a 10 x 10 x num_channels matrix."""

        if self._kwargs["env_type"] == "minatar":
            return SFNetworkMinatar(
                num_actions=self._kwargs["action_shape"],
                sf_dim=self._kwargs["sf_dim"],
                feature_dim=self._kwargs["feature_dim"],
            )

        else:
            return SFNetwork(
                num_actions=self._kwargs["action_shape"],
                sf_dim=self._kwargs["sf_dim"],
                hidden_dim=self._kwargs["hidden_dim"],
            )

    def get_state(self) -> Mapping[Text, Any]:
        """Retrieves agent state as a dictionary (e.g. for serialization)."""
        state = {
            "rng_key": self._rng_key,
            "frame_t": self._frame_t,
            "frame_t_exploration": self._frame_t_exploration,
            "opt_state": self._opt_state,
            "online_params": self._online_params,
            "target_params": self._target_params,
            "replay": self._replay.get_state(),
        }
        return state

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets agent state from a (potentially de-serialized) dictionary."""
        self._rng_key = state["rng_key"]
        self._frame_t = state["frame_t"]
        self._frame_t_exploration = state["frame_t_exploration"]
        self._opt_state = jax.device_put(state["opt_state"])
        self._online_params = jax.device_put(state["online_params"])
        self._target_params = jax.device_put(state["target_params"])
        self._replay.set_state(state["replay"])

    def get_rep_sf_and_val(self, timestep: dm_env.TimeStep, task: Array):
        timestep = self._preprocessor(timestep)
        s_t = timestep.observation
        output = self._get_output(
            rng_key=self._eval_rng_key,
            online_params=self._online_params,
            s_t=s_t,
            task=task,
            network=self._network,
        )
        q_t_1_u1 = jnp.take(output.q_1, indices=0, axis=0)
        sf_t = output.sf[0]  # (action_dim, sf_dim,)
        qval_t = q_t_1_u1
        obs_t = output.basis_features[0]
        v_t = jnp.max(qval_t, axis=-1)

        return sf_t, v_t, obs_t, qval_t

    def get_sf(self, timestep: dm_env.TimeStep, task: Array):
        timestep = self._preprocessor(timestep)
        s_t = timestep.observation
        output = self._get_output(
            rng_key=self._eval_rng_key,
            online_params=self._online_params,
            s_t=s_t,
            task=task,
            network=self._network,
        )
        sf_1 = output.sf[0]
        return sf_1

    def get_qval(self, timestep: dm_env.TimeStep, task: Array) -> Array:
        timestep = self._preprocessor(timestep)
        s_t = timestep.observation
        output = self._get_output(
            rng_key=self._eval_rng_key,
            online_params=self._online_params,
            s_t=s_t,
            task=task,
            network=self._network,
        )
        return output.q_1[0]

    @property
    def eval_network(self) -> Network:
        return self._network_eval

    @property
    def online_params(self) -> Params:
        return self._online_params

    def reset_optimizer(self) -> None:
        """
        Reset the optimizers
        Returns
        -------
        None
        """
        logging.log(logging.INFO, "Reset Optimizer")

        self._opt_state = self._optimizer.init(self._online_params)
        self._opt_state_task = self._optimizer_task.init(self._task_params)

    @property
    def importance_sampling_exponent(self) -> float:
        """Returns current importance sampling exponent of prioritized replay."""
        assert self._kwargs["use_priority_replay"]
        return self._replay.importance_sampling_exponent

    @property
    def max_seen_priority(self) -> float:
        """Returns maximum seen replay priority up until this time."""
        assert self._kwargs["use_priority_replay"]
        return self._max_seen_priority

    @staticmethod
    def del_unwanted_items_from_metric(metric) -> dict[str, Any]:
        if "opt_state_task" in metric:
            del metric["opt_state_task"]

        if "opt_state_task_consolidation" in metric:
            del metric["opt_state_task_consolidation"]

        if "task_params" in metric:
            del metric["task_params"]

        if "rng_key" in metric:
            del metric["rng_key"]

        return metric

    def get_transition_accumulator(
        self,
    ) -> Union[TransitionTaskAccumulator, NStepTransitionTaskAccumulator,]:
        # create transition accumulator
        if self._kwargs["nstep"] > 1:
            return NStepTransitionTaskAccumulator(
                n=self._kwargs["nstep"], num_actions=self._kwargs["action_shape"]
            )

        else:
            return TransitionTaskAccumulator(num_actions=self._kwargs["action_shape"])

    def get_replay_structure(self) -> ReplayStructure:
        return Transition_Task(
            s_tm1=None,
            a_tm1=None,
            r_t=None,
            discount_t=None,
            s_t=None,
            task=None,
        )

    @online_params.setter
    def online_params(self, network_params: Params) -> None:
        logging.log(logging.INFO, "Network params for eval set")
        self._online_params = network_params

    @property
    def num_params(self) -> int:
        """Returns the number of parameters in the network."""
        return sum(x.size for x in jax.tree_util.tree_leaves(self._online_params))

    @property
    def consolidation(self) -> bool:
        return self._kwargs["consolidation"]

    @property
    def use_plasticity_injection(self) -> bool:
        return self._kwargs["use_plasticity_injection"]
