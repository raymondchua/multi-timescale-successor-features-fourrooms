import jax
import rlax
from typing import Unpack, Callable, Tuple, Any, Text, Mapping, Union
import chex
import distrax
import optax
import dm_env
import pickle

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
from network import QNetworkMinatar, QNetOutputs, DQNetwork
import jax.numpy as jnp
from functools import partial
from helpers import (
    processors,
    ReplayStructure,
    TransitionAccumulator,
    NStepTransitionAccumulator,
)
from absl import logging
from pathlib import Path
from collections import OrderedDict


# Batch variant of q_learning.
_batch_q_learning = jax.vmap(rlax.q_learning)
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)


class DQNAgentKwargs(BaseAgentKwargs):
    """Keyword arguments for DQN agent."""

    consolidation: bool
    feature_dim: int
    use_plasticity_injection: bool


class DQNAgent(BaseAgent):
    def __init__(self, **kwargs: Unpack[DQNAgentKwargs]):
        super().__init__(**kwargs)

        self._kwargs = kwargs
        self._network = self.get_network_fn()
        self._network_eval = self._network

        self._online_params = self._network.init(
            self._network_rng_key, self._sample_network_input_extended
        )["params"]
        self._target_params = self._online_params
        self._opt_state = self._optimizer.init(self._online_params)

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
            return DQNetwork(
                feature_dim=self._kwargs["feature_dim"],
                hidden_dim=self._kwargs["hidden_dim"],
                num_actions=self._kwargs["action_shape"],
            )

    @staticmethod
    @partial(jax.jit, static_argnames="network")
    def _get_output(
        rng_key: PRNGKey,
        online_critic_sf_params: Params,
        s_t: Array,
        network: Network,
    ) -> QNetOutputs:
        """Forward pass through the network for inference. No gradient."""
        _, apply_key = jax.random.split(rng_key)
        output = network.apply(
            {"params": online_critic_sf_params},
            s_t[None, ...],
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

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """
        self._transition_accumulator.reset()
        processors.reset(self._preprocessor)
        self._action = None

    def _act(self, timestep: dm_env.TimeStep) -> Action:
        """Selects action given timestep, according to epsilon-greedy policy."""
        s_t = timestep.observation
        self._rng_key, a_t, v_t = self._select_action(
            rng_key=self._rng_key,
            network_params=self._online_params,
            s_t=s_t,
            exploration_epsilon=self.exploration_epsilon,
            network=self._network,
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

    def reset_replay_buffer(self) -> None:
        """Reset the replay buffer of the agent"""
        logging.log(logging.INFO, "Reset Replay buffer")
        self._replay.reset()

    def save_snapshot(
        self, task_id: int, snapshot_dir: str, step: int, cfg: dict = None
    ) -> None:
        params = jax.device_get(self.online_params)
        snapshot_dir = self._kwargs["work_dir"] / Path(snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_task{task_id}.pkl"
        with snapshot.open("wb") as f:
            pickle.dump(params, f)

        # log snapshot saved for task and step
        logging.info("Snapshot saved for task %d and step %d", task_id, step)

        if cfg is not None:
            # save the config file as well if it has not been saved before
            cfg_file = snapshot_dir / f"cfg.pkl"

            if cfg_file.exists():
                logging.info("Config file already exists in snapshot directory")

            else:
                with cfg_file.open("wb") as f:
                    pickle.dump(cfg, f)
                logging.info("Config file saved in snapshot directory")

    def load_snapshot(self, task_id: int, snapshot_dir: str) -> None:
        snapshot_dir = Path(snapshot_dir)

        # check if snapshot_dir exists, if not raise error
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot directory {snapshot_dir} does not exist")

        snapshot = snapshot_dir / f"snapshot_task{task_id}.pkl"
        with snapshot.open("rb") as f:
            params = pickle.load(f)
            self._online_params = jax.device_put(params)
            self._target_params = self._online_params

    @property
    def eval_network(self) -> Network:
        return self._network_eval

    @property
    def online_params(self) -> Params:
        return self._online_params

    @property
    def solved_meta(self):
        return OrderedDict()

    def get_rep_and_val(self, timestep: dm_env.TimeStep):
        timestep = self._preprocessor(timestep)
        s_t = timestep.observation
        output = self._get_output(
            rng_key=self._eval_rng_key,
            online_critic_sf_params=self._online_params,
            s_t=s_t,
            network=self._network,
        )
        qval_t = output.q_1[0]
        obs_t = output.obs_rep[0]
        v_t = jnp.max(qval_t, axis=-1)

        return (
            v_t,
            obs_t,
            qval_t,
        )

    def reset_optimizer(self) -> None:
        """
        Reset the optimizers
        Returns
        -------
        None
        """
        logging.log(logging.INFO, "Reset Optimizer")
        self._opt_state = self._optimizer.init(self._online_params)

    def get_transition_accumulator(
        self,
    ) -> Union[NStepTransitionAccumulator, TransitionAccumulator]:
        # create transition accumulator
        if self._kwargs["nstep"] > 1:
            return NStepTransitionAccumulator(n=self._kwargs["nstep"])

        else:
            return TransitionAccumulator()

    def get_replay_structure(self) -> ReplayStructure:
        return Transition(
            s_tm1=None,
            a_tm1=None,
            r_t=None,
            discount_t=None,
            s_t=None,
        )

    @online_params.setter
    def online_params(self, network_params: Params) -> None:
        logging.log(logging.INFO, "Network params for eval set")
        self._online_params = network_params

    @property
    def consolidation(self) -> bool:
        return self._kwargs["consolidation"]

    @property
    def use_plasticity_injection(self) -> bool:
        return self._kwargs["use_plasticity_injection"]
