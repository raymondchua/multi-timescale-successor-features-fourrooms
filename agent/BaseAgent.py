from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Text,
    Tuple,
    TypeVar,
    TypedDict,
    Union,
)
import dm_env
import numpy as np
import sys
from typing_extensions import Unpack, Literal
import chex
from helpers.td_error_scaler_jax import (
    init_td_error_scaler_state,
    update_td_error_scaler_state,
    compute_sigma,
)

from functools import partial

import logging
from flax.core import FrozenDict
import flax

Action = int
Params = FrozenDict
Network = flax.linen.Module
NetworkFn = Callable[..., Any]
PRNGKey = jnp.ndarray  # A size 2 array.

Array = chex.Array
Numeric = chex.Numeric

from helpers import (
    compress_array,
    uncompress_array,
    optimizers,
    LinearSchedule,
    NStepTransitionAccumulator,
    NStepTransitionTaskAccumulator,
    PrioritizedTransitionReplay,
    ReplayStructure,
    TransitionAccumulator,
    TransitionTaskAccumulator,
    Transition,
    TransitionReplay,
    Transition_Task,
)


def make_actions_vector(num_actions):
    action_inputs = None

    for action in range(num_actions):
        one_hot = jnp.zeros(num_actions)
        one_hot = one_hot.at[action].set(1)
        if action_inputs is None:
            action_inputs = one_hot
        else:
            if action_inputs.ndim == 1:
                action_inputs = jnp.stack((action_inputs, one_hot))
            else:
                action_inputs = jnp.concatenate(
                    (action_inputs, jnp.expand_dims(one_hot, axis=0))
                )

    return action_inputs


class BaseAgentKwargs(TypedDict):
    action_repeat: int
    action_shape: int
    batch_size: int
    clip_reward: bool
    critic_target_tau: float
    discount: float
    double_q_learning: bool
    env_type: str
    environment_height: int
    environment_width: int
    eval_exploration_epsilon: float
    exploration_epsilon_begin_value: Union[float, int]
    exploration_epsilon_decay_frame_fraction: float
    exploration_epsilon_decay_steps: int
    exploration_epsilon_end_value: float
    num_stacked_frames: int
    grad_error_bound: Union[int, float]
    has_attention_mechanism: bool
    hidden_dim: int
    impala_conv_scale: int
    importance_sampling_exponent_begin_value: float
    importance_sampling_exponent_end_value: float
    log_grads: bool
    lr: float
    max_seen_priority: float
    min_replay_capacity_fraction: float
    name: str
    nstep: int
    normalize_weights: bool
    num_train_frames: int
    obs_shape: tuple[int, ...]
    obs_type: str
    optimizer: str
    optimizer_b1: float
    optimizer_b2: float
    policy: str
    priority_exponent: float
    print_metrics: bool
    replay_buffer_size: int
    scale_reward: bool
    seed: int
    target_network_update_period: int
    use_impala_encoder: bool
    use_preprocessor: bool
    use_priority_replay: bool
    uniform_sample_probability: float
    use_double_q: bool
    use_soft_target: bool
    use_td_scaling: bool
    use_wandb: bool
    work_dir: str


AgentKwargsType = TypeVar("AgentKwargsType", bound=BaseAgentKwargs)


class BaseAgent(ABC):
    def __init__(self, **kwargs: Unpack[BaseAgentKwargs]):
        self._meta_dim = 0
        self._max_seen_priority = 1.0
        self._kwargs = kwargs
        self._g = 0.0
        self._td_scaler_state = init_td_error_scaler_state()

        self._random_state = np.random.RandomState(self._kwargs["seed"])

        self._rng_key = jax.random.PRNGKey(
            self._random_state.randint(
                -sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64
            )
        )

        self._train_rng_key, self._eval_rng_key = jax.random.split(self._rng_key)
        self._train_rng_key, self._network_rng_key = jax.random.split(
            self._train_rng_key
        )

        self._min_replay_capacity = (
            self._kwargs["min_replay_capacity_fraction"]
            * self._kwargs["replay_buffer_size"]
        )

        self._action_inputs = make_actions_vector(kwargs["action_shape"])
        self._preprocessor = None

        # Other agent state: last action, frame count, etc.
        self._action = None
        self._frame_t = -1  # Current frame index.
        self._frame_t_exploration = -1
        self._statistics = {"state_value": np.nan}
        self._preprocessors_list = []

        self._exploration_epsilon = LinearSchedule(
            begin_t=int(
                self._kwargs["min_replay_capacity_fraction"]
                * self._kwargs["replay_buffer_size"]
                * self._kwargs["action_repeat"]
            ),
            decay_steps=self._kwargs["exploration_epsilon_decay_steps"],
            begin_value=self._kwargs["exploration_epsilon_begin_value"],
            end_value=self._kwargs["exploration_epsilon_end_value"],
        )

        self._transition_accumulator = self.get_transition_accumulator()
        compress_encoder, compress_decoder = self.get_encoder_decoder()

        self._replay_structure = self.get_replay_structure()

        self._replay = self.replay_buffer(
            replay_buffer_size=self._kwargs["replay_buffer_size"],
            random_state=self._random_state,
            compress_encoder=compress_encoder,
            compress_decoder=compress_decoder,
            replay_structure=self._replay_structure,
        )

        self._optimizer = optimizers.get_optimizer(
            self._kwargs["optimizer"],
            self._kwargs["lr"],
            self._kwargs["optimizer_b_1"],
            self._kwargs["optimizer_b_2"],
        )

        sample_network_input = jnp.zeros(
            (
                self._kwargs["environment_height"],
                self._kwargs["environment_width"],
                self._kwargs["num_stacked_frames"],
            )
        )
        self._sample_network_input_extended = sample_network_input[None, ...]

        # Initialize actor critic network parameters and optimizer.
        print(
            "sample_network_input_extended shape: ",
            self._sample_network_input_extended.shape,
        )

    @staticmethod
    def loss_fn(
        online_params: Params,
        target_params: Params,
        transitions: Union[Transition, Transition_Task],
        rng_key: PRNGKey,
        network: Network,
        grad_error_bound: float,
        batch_size: int,
        weights: Array,
        td_error_scaler_sigma: Array,
    ) -> [Array, Array]:
        """Computes the standard td loss"""
        ...

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
    ) -> [Array, Array]:
        """Computes the double q loss"""
        ...

    @staticmethod
    @abstractmethod
    def _select_action(
        rng_key: PRNGKey,
        network_params: Params,
        s_t: Array,
        exploration_epsilon: float,
        network: Network,
    ) -> Tuple[PRNGKey, Array, Array]:
        ...

    @staticmethod
    @abstractmethod
    def _update(
        rng_key: PRNGKey,
    ):
        ...

    @abstractmethod
    def get_network_fn(self) -> Network:
        ...

    @property
    @abstractmethod
    def online_params(self) -> Params:
        ...

    @online_params.setter
    @abstractmethod
    def online_params(self, network_params: Params) -> None:
        ...

    @property
    def eval_rng_key(self) -> PRNGKey:
        return self._eval_rng_key

    # Might be needed for tabular RL agents
    def compute_loss(self, *argv):
        raise NotImplementedError

    def preprocessors(self, preprocessor: Callable):
        assert preprocessor is not None, "Preprocessor cannot be None"
        self._preprocessor = preprocessor

    def get_encoder_decoder(self) -> Tuple[Callable, Callable]:
        """Returns the encoder and decoder functions for the replay buffer"""
        if self._kwargs["compress_state"]:

            def compress_encoder(transition):
                return transition._replace(
                    s_tm1=compress_array(transition.s_tm1),
                    s_t=compress_array(transition.s_t),
                )

            def compress_decoder(transition):
                return transition._replace(
                    s_tm1=uncompress_array(transition.s_tm1),
                    s_t=uncompress_array(transition.s_t),
                )

        else:
            compress_encoder = None
            compress_decoder = None

        return compress_encoder, compress_decoder

    @abstractmethod
    def get_transition_accumulator(
        self,
    ) -> Union[
        TransitionAccumulator,
        NStepTransitionAccumulator,
        TransitionTaskAccumulator,
        NStepTransitionTaskAccumulator,
    ]:
        ...

    def replay_buffer(
        self,
        replay_buffer_size: int,
        random_state: np.random.RandomState,
        compress_encoder: Callable,
        compress_decoder: Callable,
        replay_structure: ReplayStructure,
    ) -> Union[TransitionReplay, PrioritizedTransitionReplay]:

        if self._kwargs["use_priority_replay"]:
            priority_exponent = self._kwargs["priority_exponent"]
            importance_sampling_exponent_schedule = (
                self.importance_sampling_exponent_schedule()
            )
            uniform_sample_probability = self._kwargs["uniform_sample_probability"]
            normalize_weights = self._kwargs["normalize_weights"]
            replay = PrioritizedTransitionReplay(
                replay_buffer_size,
                replay_structure,
                priority_exponent,
                importance_sampling_exponent_schedule,
                uniform_sample_probability,
                normalize_weights,
                random_state,
                compress_encoder,
                compress_decoder,
            )

        else:
            replay = TransitionReplay(
                replay_buffer_size,
                replay_structure,
                random_state,
                compress_encoder,
                compress_decoder,
            )
        return replay

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        # Check for DeviceArrays in values as this can be very slow.
        # assert all(
        #     not isinstance(x, jnp.DeviceArray) for x in self._statistics.values()
        # )

        # Change for compatibility with JAX 0.4.25
        # Convert JAX arrays to Python-native types
        self._statistics = {k: (v.item() if isinstance(v, jax.Array) and v.ndim == 0 else jax.device_get(v))
                            for k, v in self._statistics.items()}
        assert all(not isinstance(x, jax.Array) for x in self._statistics.values())
        return self._statistics

    @property
    def exploration_epsilon(self) -> float:
        """Returns epsilon value currently used by (eps-greedy) behavior policy."""
        return self._exploration_epsilon(self._frame_t_exploration)

    @property
    def eval_exploration_epsilon(self) -> float:
        """Returns epsilon value currently used by (eps-greedy) behavior policy."""
        return self._kwargs["eval_exploration_epsilon"]

    @property
    def consolidation(self) -> bool:
        """Returns the consolidation flag if the agent uses the consolidation mechanism."""
        raise NotImplementedError

    @property
    def use_plasticity_injection(self) -> bool:
        """Returns whether the agent uses the plasticity injection mechanism."""
        raise NotImplementedError

    def _log_weights_and_biases(self, data, name) -> dict[str, float]:
        """Logs the weights and biases of the network."""

        metrics = dict()

        for row, module in enumerate(sorted(data)):

            if "w" not in data[module] and "b" not in data[module]:
                continue

            weights_name = "train/" + f"{module}_" + name + "/w"
            metrics[weights_name] = data[module]["w"]

            if "b" in data[module]:
                bias_name = "train/" + f"{module}_" + name + "/b"
                metrics[bias_name] = data[module]["b"]

        return metrics

    def reset_exploration_frame_counter(self) -> None:
        """Reset agent's internal frame counter to reset the epsilon greedy exploration"""
        logging.log(logging.INFO, "Reset exploration frame counter")
        self._frame_t_exploration = -1

    @property
    @abstractmethod
    def eval_network(self) -> Network:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat"""
        ...

    @abstractmethod
    def step(
        self,
        timestep: dm_env.TimeStep,
        time_to_learn: bool,
        **kwargs,
    ) -> dict[str, Any]:
        """Selects action given timestep and potentially learns."""
        ...

    @abstractmethod
    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets the agent's state."""
        ...

    @staticmethod
    def convert_variable_into_batch(variable, batch_size: int) -> Array:
        var_batch = jnp.tile(variable, batch_size)
        var_batch = jnp.reshape(var_batch, (batch_size, -1))
        return var_batch

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

    def importance_sampling_exponent_schedule(self):
        return LinearSchedule(
            begin_t=int(
                self._kwargs["min_replay_capacity_fraction"]
                * self._kwargs["replay_buffer_size"]
            ),
            end_t=(
                int(self._kwargs["num_train_frames"] / self._kwargs["action_repeat"])
            ),
            begin_value=self._kwargs["importance_sampling_exponent_begin_value"],
            end_value=self._kwargs["importance_sampling_exponent_end_value"],
        )

    @abstractmethod
    def get_replay_structure(self) -> ReplayStructure:
        ...

    def update_priorities(self, indices: list[int], td_errors: Array) -> None:
        assert self._kwargs["use_priority_replay"]
        priorities = jnp.abs(td_errors)
        priorities = jax.device_get(priorities)
        max_priority = jnp.max(priorities)
        self._max_seen_priority = max(self._max_seen_priority, max_priority)
        self._replay.update_priorities(indices, priorities)

    @staticmethod
    def log_grads_norm(metrics: dict, gradients: dict, name: str) -> dict[str, Any]:
        """Logs the norm of the gradients of the network."""
        assert gradients is not None

        for key, value in gradients.items():
            # compute the norm of the gradients only if value has a key named "w" or "b" for weights and biases
            if isinstance(value, dict):
                if "w" in value.keys():
                    metrics[
                        "{name}_{key}_w grad norm".format(name=name, key=key)
                    ] = jnp.linalg.norm(value["w"]).item()

                elif "b" in value.keys():
                    metrics[
                        "{name}_{key}_b grad norm".format(name=name, key=key)
                    ] = jnp.linalg.norm(value["b"]).item()
            else:
                # for the gain in attention: sf_keys_gain, sf_values_gain
                metrics[f"{name}_{key} grad norm"] = jnp.linalg.norm(value).item()

        return metrics

    @staticmethod
    def log_params_norm(metrics: dict, params: dict, name: str) -> dict[str, Any]:
        """Logs the norm of the parameters of the network."""
        assert params is not None

        for key, value in params.items():
            # compute the norm of the gradients only if value has a key named "w" or "b" for weights and biases
            if "w" in value.keys():
                metrics[
                    "{name}_{key}_w norm".format(name=name, key=key)
                ] = jnp.linalg.norm(value["w"]).item()

                if key == "q_u1" or key == "q_u2" or key == "q_u3":
                    metrics[
                        "{name}_{key}_w action_0_weight_0".format(name=name, key=key)
                    ] = value["w"][0, 0].item()

            elif "b" in value.keys():
                metrics[
                    "{name}_{key}_b norm".format(name=name, key=key)
                ] = jnp.linalg.norm(value["b"]).item()

                if key == "q_u1" or key == "q_u2" or key == "q_u3":
                    metrics[
                        "{name}_{key}_b action_0_weight_0".format(name=name, key=key)
                    ] = value["b"][0].item()

        return metrics

    @property
    def num_params(self) -> int:
        """Returns the number of parameters in the network."""
        return sum(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda x: x.size, self.online_params)
            )
        )

    @property
    def g(self) -> float:
        """Returns the returns."""
        return self._g

    def add_to_g(self, reward) -> None:
        """Add reward to the returns."""
        self._g += reward

    def reset_G(self) -> None:
        """Reset the returns."""
        self._g = 0

    @partial(jax.jit, static_argnums=(0,))
    def _update_td_scaler_state(self, state, reward, discount, returns):
        return update_td_error_scaler_state(state, reward, discount, returns)

    def update_td_scaler(self, reward, discount, returns) -> None:
        self._td_scaler_state = self._update_td_scaler_state(
            self._td_scaler_state, reward, discount, returns
        )

    @property
    def td_sigma(self):
        return compute_sigma(self._td_scaler_state)

    @property
    def has_attention_mechanism(self) -> bool:
        return self._kwargs["has_attention_mechanism"]

    @property
    def name(self) -> str:
        return self._kwargs["name"]


