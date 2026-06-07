from typing import Any, Callable, Mapping, Text, Tuple, Sequence, List

import jax
import jax.numpy as jnp
import numpy as np
import chex
import dm_env
from flax.core import freeze, unfreeze
import pickle
from pathlib import Path

from functools import partial

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

from typing_extensions import Unpack
from dm_env import specs
from jax import random

from losses import update_and_accumulate_tree, pytree_l2_norm
from network import SFNetwork, SFNetOutputs

from helpers.successor_features import get_all_sf

from helpers import optimizers

from collections import OrderedDict

from absl import logging


class SFConsolidationParamsContinuousAgentKwargs(SimpleSFAgentKwargs):
    beaker_capacity: int
    lr_consolidation: float
    num_beakers: int
    optimizer_consolidation: str
    flow_factor: int
    update_consolidation_every_steps: int
    consolidation: bool


class SFConsolidationParamsContinuousAgent(SimpleSFAgent):
    def __init__(
        self,
        **kwargs: Unpack[SFConsolidationParamsContinuousAgentKwargs],
    ):

        super().__init__(**kwargs)
        self._beaker_capacity = self._kwargs["beaker_capacity"]
        self._consolidation = self._kwargs["consolidation"]
        self._flow_factor = self._kwargs["flow_factor"]
        self._num_beakers = self._kwargs["num_beakers"]
        self._lr_consolidation = self._kwargs["lr_consolidation"]
        self._update_consolidation_every_steps = self._kwargs["update_consolidation_every_steps"]

        self._capacity = jnp.zeros(self._num_beakers + 1, dtype=jnp.int32) # add one for t
        self._num_train_frames = None

        # self._networks = []
        self._network = self.get_network_fn()
        self._network_params = []

        for exp in range(self._num_beakers):

            if exp == 0:
                self._capacity = self._capacity.at[exp].set(1)

            else:
                self._capacity = self._capacity.at[exp].set((self._beaker_capacity ** exp) * self._flow_factor)

            self._rng_key, rng_n = jax.random.split(self._rng_key)
            current_network_params = self._network.init(
                rng_n,
                self._sample_network_input_extended,
                self._sample_task_input_extended,
            )["params"]

            # self._networks.append(current_network)
            self._network_params.append(current_network_params)

        self._capacity = self._capacity.at[self._num_beakers].set((self._beaker_capacity ** self._num_beakers) * self._flow_factor)

        self._mask = None
        self._params_set_to_zero = unfreeze(
            jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x), unfreeze(self._network_params[0])
            )
        )

        self._online_params = self._network_params[0]
        self._target_params = self._online_params
        self._opt_state = self._optimizer.init(self._online_params)

    # Define jitted loss, update, and policy functions as static methods,
    # to emphasize that these are meant to be pure functions
    # and should not access the agent object's state via `self`.

    def set_up_consolidation_system(self):
        self._g_flow = 0.1 / self._capacity[1]
        self._storage_timescales = jnp.zeros(
            self._num_beakers, dtype=jnp.int32
        )
        self._recall_timescales = jnp.zeros(
            self._num_beakers, dtype=jnp.int32
        )

        self._scale_consolidation = jnp.zeros(
            self._num_beakers, dtype=jnp.float32
        )

        self._scale_recall = jnp.zeros(
            self._num_beakers, dtype=jnp.float32
        )

        for exp in range(self._num_beakers):
            self._storage_timescales = self._storage_timescales.at[exp].set(jnp.ceil(
                (
                        self._capacity[exp]
                        / (self._g_flow * self._lr_consolidation)
                ) * self._update_consolidation_every_steps
            ))

            self._recall_timescales = self._recall_timescales.at[exp].set(self._storage_timescales[exp])

            self._scale_consolidation = self._scale_consolidation.at[exp].set(self._g_flow / self._capacity[exp])
            self._scale_recall = self._scale_recall.at[exp].set(self._g_flow / self._capacity[exp + 1])


        logging.info(f"g_flow: {self._g_flow}")
        logging.info(f"Capacity: {self._capacity}")
        logging.info(f"storage g_flow: {self._g_flow}")
        logging.info(f"recall g_flow: {self._g_flow}")
        logging.info(f"storage timescales: {self._storage_timescales}")
        logging.info(f"recall timescales: {self._recall_timescales}")
        logging.info(f"scale consolidation: {self._scale_consolidation}")
        logging.info(f"scale recall: {self._scale_recall}")

        assert (
                self._g_flow * self._update_consolidation_every_steps <= 0.1
        ), "g_flow * update_consolidation_every_steps should be less than or equal to 0.1"

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
        loss = 0.0
        params_norm = jnp.zeros(num_beakers)

        # Stack list of PyTrees into a PyTree of arrays
        params_stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *params)

        def get_beaker(ps, i):
            return jax.tree_util.tree_map(lambda x: x[i], ps)

        def set_beaker(ps, i, new_p):
            return jax.tree_util.tree_map(lambda x, new_x: jax.lax.dynamic_update_index_in_dim(x, new_x, i, axis=0), ps, new_p)

        # First beaker
        p0 = get_beaker(params_stacked, 0)
        p1 = get_beaker(params_stacked, 1)
        scale_first = g_flow / capacity[1]
        p0, loss = update_and_accumulate_tree(p0, p1, scale_first * lr_consolidation * mask[1], loss)
        params_stacked = set_beaker(params_stacked, 0, p0)
        params_norm = params_norm.at[0].set(pytree_l2_norm(p0))

        def scan_body_fn(carry, i):
            ps, loss = carry
            p_prev = get_beaker(ps, i - 1)
            p_next = get_beaker(ps, i + 1)
            p_i = get_beaker(ps, i)

            scale_prev = g_flow / capacity[i]
            scale_next = g_flow / capacity[i + 1]

            # Consolidate from previous
            p_i, loss = update_and_accumulate_tree(p_i, p_prev, scale_prev * update_consolidation_every_steps, loss)

            # Recall from next
            def do_recall(p, l):
                return update_and_accumulate_tree(p, p_next, scale_next * lr_consolidation, l)

            def no_recall(p, l):
                return p, l

            p_i, loss = jax.lax.cond(mask[i] != 0, do_recall, no_recall, p_i, loss)
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
        mask = self.compute_recall_mask(self._update_step, self._recall_timescales)

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
            learn_metrics = self._learn(mask=mask, update_meta=True, update_consolidation=update_consolidation)

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
            learn_metrics = self._learn(mask=mask, update_meta=False, update_consolidation=update_consolidation)

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

        if self._kwargs["print_metrics"]:
            for key, value in metrics.items():
                if key != "meta":
                    print(f"{key}: {value}")

        return metrics

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
            loss_fn = self.double_q_loss_fn
        else:
            loss_fn = self.loss_fn

        reward_from_env = transitions.r_t

        metrics = dict()

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

            self._online_params = self._network_params[0]
            metrics["consolidation_loss"] = consolidation_loss

            for i in range(self._kwargs["num_beakers"]):
                metrics[f"params_u{i}_norm"] = network_params_norm[i]

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

        sf_all = get_all_sf(
            transitions=transitions,
            num_beakers=self._num_beakers,
            network_params=self._network_params,
            network=self._network,
        )
        sf_all = jnp.swapaxes(sf_all, 2, 3)

        chex.assert_shape(sf_all, (
            self._kwargs["batch_size"],
            self._kwargs["num_beakers"],
            self._kwargs["action_shape"],
            self._kwargs["sf_dim"],
        ))

        # Q-value computation (batch, beaker, action)
        q_val = jnp.einsum("buad,d->bua", sf_all, self._task_params["w"])
        q_val_mean = q_val.mean(axis=0)  # (num_beakers, num_actions)

        # loop over beakers and store the sf for each beaker and action
        for i in range(self._kwargs["num_beakers"]):
            for j in range(self._kwargs["action_shape"]):
                metrics[f"q_val_u{i}_a{j}_mean"] = q_val_mean[i, j]
            metrics[f"mask_u{i}"] = mask[i]

        return metrics

    def get_network_fn(self) -> Network:
        """Returns a function that computes the network output."""
        if self._kwargs["env_type"] == "minatar":
            raise NotImplementedError(
                "SFConsolidationParamsContinuousAgent does not support minatar yet."
            )

        else:
            return SFNetwork(
                num_actions=self._kwargs["action_shape"],
                sf_dim=self._kwargs["sf_dim"],
                hidden_dim=self._kwargs["hidden_dim"],
            )

    @staticmethod
    @partial(jax.jit, static_argnames="network")
    def _get_output(
        rng_key: PRNGKey,
        online_params: Params,
        s_t: Array,
        task: Array,
        network: Network,
    ) -> SFNetOutputs:
        _, apply_key = jax.random.split(rng_key)
        output = network.apply(
            {"params": online_params},
            s_t[None, ...],
            task[None, ...],
        )
        return jax.lax.stop_gradient(output)

    @property
    def online_params(self) -> Params:
        return self._online_params

    @online_params.setter
    def online_params(self, network_params: Params) -> None:
        logging.log(logging.INFO, "Network params for eval set")
        self._online_params = network_params
        self._network_params[0] = network_params

    @property
    def num_params(self) -> int:
        """Returns the number of parameters in the network."""
        return self._num_beakers * sum(
            x.size for x in jax.tree_util.tree_leaves(self._online_params)
        )

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
            self._network_params[0] = jax.device_put(params)
            self._online_params = params
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

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets agent state from a (potentially de-serialized) dictionary."""
        self._rng_key = state["rng_key"]
        self._frame_t = state["frame_t"]
        self._frame_t_exploration = state["frame_t_exploration"]
        self._opt_state = jax.device_put(state["opt_state"])
        self._network_params[0] = jax.device_put(state["online_params"])
        self._online_params = self._network_params[0]
        self._target_params = jax.device_put(state["target_params"])
        self._replay.set_state(state["replay"])

    @staticmethod
    @jax.jit
    def compute_recall_mask(update_step: int, recall_timescales: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the recall mask based on a fixed recall_timescales and the current update step.

        Args:
            update_step: scalar int
            recall_timescales: static 1D array (e.g. [0, 100, 200, ...])

        Returns:
            mask: binary (int32) mask of shape (num_beakers,)
        """
        mask = (recall_timescales < update_step).astype(jnp.int32)
        mask = jnp.concatenate([jnp.array([1], dtype=jnp.int32), mask[:-1]])
        return mask

    @property
    def num_train_frames(self) -> int:
        return self._num_train_frames

    @num_train_frames.setter
    def num_train_frames(self, num_train_frames: int):
        """
        Set the number of training frames for the agent.
        This is used to determine when to stop training.
        """
        self._num_train_frames = num_train_frames

    @property
    def consolidation(self):
        return self._consolidation
