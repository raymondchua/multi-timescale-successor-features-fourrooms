"""Composable timestep processing, for preprocessing.
Aims:
* Be self-contained.
* Easy to have the preprocessing on the agent side or on the environment side.
* Easy to swap out and modify parts of the processing.
Conventions:
* The term "processor" is used to refer to any callable that could also have
  a `reset()` function to clear any internal state. E.g. a plain function. Or an
  instance of a class with `__call__` method, with or without a `reset()`
  method.
* `None` means no output when subsampling inputs.

Above copied from DQNZOO:
https://github.com/deepmind/dqn_zoo/blob/f011d683529d8d23b017a95194ebbb41a4962fe8/dqn_zoo/processors.py
"""


import collections
from typing import Any, Callable, List, Iterable, Optional, Sequence, Text, Tuple

import chex
import dm_env
from dm_env import specs
import numpy as np
from PIL import Image

Processor = Callable  # Actually a callable that may also have a reset() method.
Nest = Any  # Recursive types are not yet supported by pytype.
NamedTuple = Any
StepType = dm_env.StepType


def reset(processor: Processor[[Any], Any]) -> None:
    """Calls `reset()` on a `Processor` or function if the method exists."""
    if hasattr(processor, "reset"):
        processor.reset()


identity = lambda v: v


def trailing_zero_pad(
    length: int,
) -> Processor[[List[np.ndarray]], List[np.ndarray]]:
    """Adds trailing zero padding to array lists to ensure a minimum length."""

    def trailing_zero_pad_fn(arrays):
        padding_length = length - len(arrays)
        if padding_length <= 0:
            return arrays
        zero = np.zeros_like(arrays[0])
        temp = arrays + [zero] * padding_length
        return arrays + [zero] * padding_length

    return trailing_zero_pad_fn

def zero_pad_to_dimension(axis: int = 3, max_channels: int = 4) -> Processor[[np.ndarray], np.ndarray]:
    """ Zero pads the input array to the given dimension such that the output channels will be increase to the same value in max_channels. """
    def zero_pad_to_dimension_fn(array):
        # if array is not a numpy array, convert it to numpy array
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # if the axis is greater than the number of dimensions in the array, raise an error
        if axis >= len(array.shape):
            raise ValueError("Axis value should be less than the number of dimensions in the array.")

        if array.shape[axis] < max_channels:
            pad_width = [(0, 0)] * len(array.shape)
            pad_width[axis] = (0, max_channels - array.shape[axis])
            array = np.pad(array, pad_width, mode='constant')
        return array
    return zero_pad_to_dimension_fn


def none_to_zero_pad(values: List[Optional[NamedTuple]]) -> List[NamedTuple]:
    """Replaces `None`s in a list of named tuples with zeros of same structure."""

    actual_values = [n for n in values if n is not None]
    if not actual_values:
        raise ValueError("Must have at least one value which is not None.")
    if len(actual_values) == len(values):
        return values
    example = actual_values[0]
    zero = type(example)(*(np.zeros_like(x) for x in example))
    return [zero if v is None else v for v in values]


def named_tuple_sequence_stack(values: Sequence[NamedTuple]) -> NamedTuple:
    """Converts a sequence of named tuples into a named tuple of tuples."""
    # [T(1, 2), T(3, 4), T(5, 6)].
    transposed = zip(*values)
    # ((1, 3, 5), (2, 4, 6)).
    return type(values[0])(*transposed)
    # T((1, 3, 5), (2, 4, 6)).


class Deque:
    """Double ended queue with a maximum length and initial values."""

    def __init__(self, max_length: int, initial_values=None):
        self._deque = collections.deque(maxlen=max_length)
        self._initial_values = initial_values or []

    def reset(self) -> None:
        self._deque.clear()
        self._deque.extend(self._initial_values)

    def __call__(self, value: Any) -> collections.deque:
        self._deque.append(value)
        return self._deque


class ApplyToNamedTupleField:
    """Runs processors on a particular field of a named tuple."""

    def __init__(self, field: Text, *processors: Processor[[Any], Any]):
        self._field = field
        self._processors = processors

    def reset(self) -> None:
        for processor in self._processors:
            reset(processor)

    def __call__(self, value: NamedTuple) -> NamedTuple:
        attr_value = getattr(value, self._field)
        for processor in self._processors:
            attr_value = processor(attr_value)

        return value._replace(**{self._field: attr_value})


class Procgen_obs_ApplyToNamedTupleField:
    """Runs processors on a particular field of a named tuple.
    Modified to handle Procgen observations which are dictionaries, which includes pixel observations, done and
    prev_level_seed value, 'prev_level_complete boolean, level_seed value.
    """

    def __init__(self, field: Text, *processors: Processor[[Any], Any]):
        self._field = field
        self._processors = processors

    def reset(self) -> None:
        for processor in self._processors:
            reset(processor)

    def __call__(self, value: NamedTuple) -> NamedTuple:
        attr_value = getattr(value, self._field)

        # if the field is observation which happens to be a dictionary, then we need to apply the processors to the
        # 'pixels' field of the observation
        if "pixels" in attr_value:
            attr_value = attr_value["pixels"]

        for processor in self._processors:
            attr_value = processor(attr_value)

        return value._replace(**{self._field: attr_value})


class Sequential:
    """Chains together multiple processors."""

    def __init__(self, *processors: Processor[[Any], Any]):
        self._processors = processors

    def reset(self) -> None:
        for processor in self._processors:
            reset(processor)

    def __call__(self, value: Any) -> Any:
        for processor in self._processors:
            value = processor(value)
        return value


class ZeroDiscountOnLifeLoss:
    """Sets discount to zero on timestep if number of lives has decreased.

    This processor assumes observations to be tuples whose second entry is a
    scalar indicating the remaining number of lives.
    """

    def __init__(self):
        self._num_lives_on_prev_step = None

    def reset(self) -> None:
        self._num_lives_on_prev_step = None

    def __call__(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        # We have a life loss when the timestep is a regular transition and lives
        # have decreased since the previous timestep.
        num_lives = timestep.observation["lives"]
        life_lost = timestep.mid() and (num_lives < self._num_lives_on_prev_step)
        self._num_lives_on_prev_step = num_lives
        return timestep._replace(discount=0.0) if life_lost else timestep


def reduce_step_type(step_types: Sequence[StepType], debug: bool = False) -> StepType:
    """Outputs a representative step type from an array of step types."""
    # Zero padding will appear to be FIRST. Padding should only be seen before the
    # FIRST (e.g. 000F) or after LAST (e.g. ML00).
    if debug:
        np_step_types = np.array(step_types)
    output_step_type = StepType.MID
    for i, step_type in enumerate(step_types):
        if step_type == 0:  # step_type not actually FIRST, but we do expect 000F.
            if debug and not (np_step_types == 0).all():
                raise ValueError("Expected zero padding followed by FIRST.")
            output_step_type = StepType.FIRST
            break
        elif step_type == StepType.LAST:
            output_step_type = StepType.LAST
            if debug and not (np_step_types[i + 1 :] == 0).all():
                raise ValueError("Expected LAST to be followed by zero padding.")
            break
        else:
            if step_type != StepType.MID:
                raise ValueError("Expected MID if not FIRST or LAST.")
    return output_step_type


class FixedPaddedBuffer:
    """Fixed size `None`-padded buffer which is cleared after it is filled.

    E.g. with `length = 3`, `initial_index = 2` and values `[0, 1, 2, 3, 4, 5, 6]`
    this will return `~~0`, `1~~`, `12~`, `123`, `4~~`, `45~`, `456`, where `~`
    represents `None`. Used to concatenate timesteps for action repeats.

    Action repeat requirements are:
    * Fixed size buffer of timesteps.
    * The `FIRST` timestep should return immediately to get the first action of
      the episode, as there is no preceding action to repeat. Prefix with padding.
    * For `MID` timesteps, the timestep buffer is periodically returned when full.
    * When a `LAST` timestep is encountered, the current buffer of timesteps is
      returned, suffixed with padding, as buffers should not cross episode
      boundaries.

    The requirements can be fulfilled by conditionally subsampling the output of
    this processor.
    """

    def __init__(self, length: int, initial_index: int):
        self._length = length
        self._initial_index = initial_index % length

        self._index = self._initial_index
        self._buffer = [None] * self._length

    def reset(self) -> None:
        self._index = self._initial_index
        self._buffer = [None] * self._length

    def __call__(self, value: Any) -> Sequence[Any]:
        if self._index >= self._length:
            assert self._index == self._length
            self._index = 0
            self._buffer = [None] * self._length
        self._buffer[self._index] = value
        self._index += 1
        return self._buffer


class ConditionallySubsample:
    """Conditionally passes through input, returning `None` otherwise."""

    def __init__(self, condition: Processor[[Any], bool]):
        self._condition = condition

    def reset(self) -> None:
        reset(self._condition)

    def __call__(self, value: Any) -> Optional[Any]:
        return value if self._condition(value) else None


class TimestepBufferCondition:
    """Returns `True` when an iterable of timesteps should be passed on.

    Specifically returns `True`:
    * If timesteps contain a `FIRST`.
    * If timesteps contain a `LAST`.
    * If number of steps passed since `FIRST` timestep modulo `period` is `0`.

    Returns `False` otherwise. Used for action repeats in Atari preprocessing.
    """

    def __init__(self, period: int):
        self._period = period
        self._steps_since_first_timestep = None
        self._should_reset = False

    def reset(self):
        self._should_reset = False
        self._steps_since_first_timestep = None

    def __call__(self, timesteps: Iterable[dm_env.TimeStep]) -> bool:
        if self._should_reset:
            raise RuntimeError("Should have reset.")

        # Find the main step type, FIRST and LAST take precedence over MID.
        main_step_type = StepType.MID
        precedent_step_types = (StepType.FIRST, StepType.LAST)
        for timestep in timesteps:
            if timestep is None:
                continue
            if timestep.step_type in precedent_step_types:
                if main_step_type in precedent_step_types:
                    raise RuntimeError("Expected at most one FIRST or LAST.")
                main_step_type = timestep.step_type

        # Must have FIRST timestep after a reset.
        if self._steps_since_first_timestep is None:
            if main_step_type != StepType.FIRST:
                raise RuntimeError("After reset first timestep should be FIRST.")

        # pytype: disable=unsupported-operands
        if main_step_type == StepType.FIRST:
            self._steps_since_first_timestep = 0
            return True
        elif main_step_type == StepType.LAST:
            self._steps_since_first_timestep = None
            self._should_reset = True
            return True
        elif (self._steps_since_first_timestep + 1) % self._period == 0:
            self._steps_since_first_timestep += 1
            return True
        else:
            self._steps_since_first_timestep += 1
            return False
        # pytype: enable=unsupported-operands


class Maybe:
    """Wraps another processor so that `None` is returned when `None` is input."""

    def __init__(self, processor: Processor[[Any], Any]):
        self._processor = processor

    def reset(self) -> None:
        reset(self._processor)

    def __call__(self, value: Optional[Any]) -> Optional[Any]:
        if value is None:
            return None
        else:
            return self._processor(value)


def rgb2y(array: np.ndarray) -> np.ndarray:
    """Converts RGB image array into grayscale."""
    chex.assert_rank(array, 3)
    output = np.tensordot(array, [0.299, 0.587, 1 - (0.299 + 0.587)], (-1, 0))
    return output.astype(np.uint8)


def resize(shape: Tuple[int, ...]) -> Processor[[np.ndarray], np.ndarray]:
    """Resizes array to the given shape."""
    if len(shape) != 2:
        raise ValueError("Resize shape has to be 2D, given: %s." % str(shape))
    # Image.resize takes (width, height) as output_shape argument.
    image_shape = (shape[1], shape[0])

    def resize_fn(array):
        image = Image.fromarray(array).resize(image_shape, Image.BILINEAR)
        return np.array(image, dtype=np.uint8)

    return resize_fn


def select_rgb_observation(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Replaces an observation tuple by its first entry (the RGB observation)."""
    return timestep._replace(observation=timestep.observation["pixels"])


def clip_reward(bound: float) -> Processor[[Optional[float]], Optional[float]]:
    """Returns a function that clips non-`None` inputs to (`-bound`, `bound`)."""

    def clip_reward_fn(reward):
        return None if reward is None else max(min(reward, bound), -bound)

    return clip_reward_fn


def scale_reward(
    min_returns: float, max_returns: float
) -> Processor[[Optional[float]], Optional[float]]:
    """Returns a function that normalize rewards between 0 and 1."""

    def scale_reward_fn(reward):
        return (
            None
            if reward is None
            else (reward - min_returns) / (max_returns - min_returns)
        )

    return scale_reward_fn


def aggregate_rewards(
    rewards: Sequence[Optional[float]], debug: bool = False
) -> Optional[float]:
    """Sums up rewards, assumes discount is 1."""
    if None in rewards:
        if debug:
            np_rewards = np.array(rewards)
            if not (np_rewards[-1] is None and (np_rewards[:-1] == 0).all()):
                # Should only ever have [0, 0, 0, None] due to zero padding.
                raise ValueError("Should only have a None reward for FIRST.")
        return None
    else:
        # Faster than np.sum for a list of floats.
        return sum(rewards)


def aggregate_discounts(
    discounts: Sequence[Optional[float]], debug: bool = False
) -> Optional[float]:
    """Aggregates array of discounts into a scalar, expects `0`, `1` or `None`."""
    if debug:
        np_discounts = np.array(discounts)
        if not np.isin(np_discounts, [0.0, 1.0, None]).all():
            raise ValueError("All discounts should be 0 or 1, got: %s." % np_discounts)
    if None in discounts:
        if debug:
            if not (np_discounts[-1] is None and (np_discounts[:-1] == 0).all()):
                # Should have [0, 0, 0, None] due to zero padding.
                raise ValueError("Should only have a None discount for FIRST.")
        return None
    else:
        # Faster than np.prod for a list of floats.
        result = 1
        for d in discounts:
            result *= d
        return result


def apply_additional_discount(additional_discount: float) -> Processor[[float], float]:
    """Returns a function that scales its non-`None` input by a constant."""
    return lambda d: None if d is None else additional_discount * d


def processor_minigrid(
    additional_discount: float = 0.99,
    max_reward: Optional[float] = 1.0,
    resize_shape: Optional[Tuple[int, int]] = (84, 84),
    use_framestack: bool = False,
    num_stacked_frames: int = 4,
) -> Processor[[dm_env.TimeStep], Optional[dm_env.TimeStep]]:
    """Standard DQN preprocessing on Atari without framestacking and action repeats"""

    # This processor does the following to a sequence of timesteps.Add framestacking if needed
    # 1. Clips rewards
    # 2. Applies an additional discount.
    # 3. Grayscales observations if frame stacking is enabled.
    # 4. Resizes observations.
    # 5. Stacks observations if frame stacking is enabled.

    if use_framestack:
        return Sequential(
            # Rewards: sum then clip.
            ApplyToNamedTupleField(
                "reward",
                clip_reward(max_reward) if max_reward else identity,
            ),
            # Discounts: take product and scale by an additional discount.
            ApplyToNamedTupleField(
                "discount",
                apply_additional_discount(additional_discount),
            ),
            # Observations: resize
            ApplyToNamedTupleField(
                "observation",
                rgb2y,
                resize(resize_shape) if resize_shape else identity,
                Deque(max_length=num_stacked_frames),
                # obs: A, ~, ~, ~, AB, ~, ~, ~, ABC, ~, ~, ~, ABCD, ~, ~, ~,
                #      BCDE, ~, ~, ~, CDEF, ...
                list,
                trailing_zero_pad(length=num_stacked_frames),
                # obs: A000, ~, ~, ~, AB00, ~, ~, ~, ABC0, ~, ~, ~, ABCD,
                #      ~, ~, ~, BCDE, ...
                lambda obs: np.stack(obs, axis=-1),
            ),
        )

    else:
        return Sequential(
            # Rewards: sum then clip.
            ApplyToNamedTupleField(
                "reward",
                clip_reward(max_reward) if max_reward else identity,
            ),
            # Discounts: take product and scale by an additional discount.
            ApplyToNamedTupleField(
                "discount",
                apply_additional_discount(additional_discount),
            ),
            # Observations: resize
            ApplyToNamedTupleField(
                "observation",
                resize(resize_shape) if resize_shape else identity,
            ),
        )


def processor_miniworld(
    additional_discount: float = 0.99,
    max_reward: Optional[float] = 1.0,
    resize_shape: Optional[Tuple[int, int]] = (84, 84),
    use_framestack: bool = False,
    num_stacked_frames: int = 4,
) -> Processor[[dm_env.TimeStep], Optional[dm_env.TimeStep]]:

    # This processor uses the same as processor_minigrid
    return processor_minigrid(
        additional_discount=additional_discount,
        max_reward=max_reward,
        resize_shape=resize_shape,
        use_framestack=use_framestack,
        num_stacked_frames=num_stacked_frames,
    )


def processor_procgen(
    clip_rwds: bool = True,
    scale_rwds: bool = False,
    additional_discount: float = 0.99,
    min_returns: Optional[float] = 0.0,
    max_returns: Optional[float] = 1.0,
    resize_shape: Optional[Tuple[int, int]] = (84, 84),
    use_framestack: bool = False,
    num_stacked_frames: int = 4,
) -> Processor[[dm_env.TimeStep], Optional[dm_env.TimeStep]]:
    """Standard DQN preprocessing on Atari but without reward clipping"""

    # This processor does the following to a sequence of timesteps.
    # 1. Clips rewards or scale rewards or do nothing.
    # 2. Applies an additional discount.
    # 3. Resizes observations.

    if use_framestack:
        if clip_rwds:
            return Sequential(
                # Rewards: sum then clip.
                ApplyToNamedTupleField(
                    "reward",
                    clip_reward(max_returns) if max_returns else identity,
                ),
                # Discounts: take product and scale by an additional discount.
                ApplyToNamedTupleField(
                    "discount",
                    apply_additional_discount(additional_discount),
                ),
                # Observations: resize
                ApplyToNamedTupleField(
                    "observation",
                    rgb2y,
                    resize(resize_shape) if resize_shape else identity,
                    Deque(max_length=num_stacked_frames),
                    # obs: A, ~, ~, ~, AB, ~, ~, ~, ABC, ~, ~, ~, ABCD, ~, ~, ~,
                    #      BCDE, ~, ~, ~, CDEF, ...
                    list,
                    trailing_zero_pad(length=num_stacked_frames),
                    # obs: A000, ~, ~, ~, AB00, ~, ~, ~, ABC0, ~, ~, ~, ABCD,
                    #      ~, ~, ~, BCDE, ...
                    lambda obs: np.stack(obs, axis=-1),
                ),
            )

        elif scale_rwds:
            return Sequential(
                # Rewards: sum then scale.
                ApplyToNamedTupleField(
                    "reward",
                    scale_reward(min_returns, max_returns) if max_returns else identity,
                ),
                # Discounts: take product and scale by an additional discount.
                ApplyToNamedTupleField(
                    "discount",
                    apply_additional_discount(additional_discount),
                ),
                # Observations: resize
                Procgen_obs_ApplyToNamedTupleField(
                    "observation",
                    rgb2y,
                    resize(resize_shape) if resize_shape else identity,
                    Deque(max_length=num_stacked_frames),
                    # obs: A, ~, ~, ~, AB, ~, ~, ~, ABC, ~, ~, ~, ABCD, ~, ~, ~,
                    #      BCDE, ~, ~, ~, CDEF, ...
                    list,
                    trailing_zero_pad(length=num_stacked_frames),
                    # obs: A000, ~, ~, ~, AB00, ~, ~, ~, ABC0, ~, ~, ~, ABCD,
                    #      ~, ~, ~, BCDE, ...
                    lambda obs: np.stack(obs, axis=-1),
                ),
            )

        else:
            return Sequential(
                # Discounts: take product and scale by an additional discount.
                ApplyToNamedTupleField(
                    "discount",
                    apply_additional_discount(additional_discount),
                ),
                # Observations: resize
                Procgen_obs_ApplyToNamedTupleField(
                    "observation",
                    rgb2y,
                    resize(resize_shape) if resize_shape else identity,
                    Deque(max_length=num_stacked_frames),
                    # obs: A, ~, ~, ~, AB, ~, ~, ~, ABC, ~, ~, ~, ABCD, ~, ~, ~,
                    #      BCDE, ~, ~, ~, CDEF, ...
                    list,
                    trailing_zero_pad(length=num_stacked_frames),
                    # obs: A000, ~, ~, ~, AB00, ~, ~, ~, ABC0, ~, ~, ~, ABCD,
                    #      ~, ~, ~, BCDE, ...
                    lambda obs: np.stack(obs, axis=-1),
                ),
            )

    else:
        if clip_rwds:
            return Sequential(
                # Rewards: sum then clip.
                ApplyToNamedTupleField(
                    "reward",
                    clip_reward(max_returns) if max_returns else identity,
                ),
                # Discounts: take product and scale by an additional discount.
                ApplyToNamedTupleField(
                    "discount",
                    apply_additional_discount(additional_discount),
                ),
                # Observations: resize
                Procgen_obs_ApplyToNamedTupleField(
                    "observation",
                    resize(resize_shape) if resize_shape else identity,
                ),
            )

        elif scale_rwds:
            return Sequential(
                # Rewards: sum then scale.
                ApplyToNamedTupleField(
                    "reward",
                    scale_reward(min_returns, max_returns) if max_returns else identity,
                ),
                # Discounts: take product and scale by an additional discount.
                ApplyToNamedTupleField(
                    "discount",
                    apply_additional_discount(additional_discount),
                ),
                # Observations: resize
                Procgen_obs_ApplyToNamedTupleField(
                    "observation",
                    resize(resize_shape) if resize_shape else identity,
                ),
            )

        else:
            return Sequential(
                # Discounts: take product and scale by an additional discount.
                ApplyToNamedTupleField(
                    "discount",
                    apply_additional_discount(additional_discount),
                ),
                # Observations: resize
                Procgen_obs_ApplyToNamedTupleField(
                    "observation",
                    resize(resize_shape) if resize_shape else identity,
                ),
            )


def processor_atari(
    additional_discount: float = 0.99,
    max_abs_reward: Optional[float] = 1.0,
    resize_shape: Optional[Tuple[int, int]] = (84, 84),
    num_action_repeats: int = 4,
    num_pooled_frames: int = 2,
    zero_discount_on_life_loss: bool = True,
    num_stacked_frames: int = 4,
    grayscaling: bool = True,
) -> Processor[[dm_env.TimeStep], Optional[dm_env.TimeStep]]:
    """Standard DQN preprocessing on Atari."""

    # This processor does the following to a sequence of timesteps.
    #
    # 1. Zeroes discount on loss of life.
    # 2. Repeats actions (previous action should be repeated if None is returned).
    # 3. Max pools action repeated observations.
    # 4. Grayscales observations.
    # 5. Resizes observations.
    # 6. Stacks observations.
    # 7. Clips rewards.
    # 8. Applies an additional discount.
    #
    # For more detail see the annotations in the processors below.

    # The FixedPaddedBuffer, ConditionallySubsample, none_to_zero_pad, stack and
    # max_pool on the observation collectively does this (step types: F = FIRST,
    # M = MID, L = LAST, ~ is None):
    #
    #   Type:      F   |   M     M     M     M   |   M     M     L   |   F   |
    #   Frames:    A   |   B     C     D     E   |   F     G     H   |   I   |
    #   Output: max[0A]|   ~     ~     ~  max[DE]|   ~     ~  max[H0]|max[0I]|
    return Sequential(
        # When the number of lives decreases, set discount to 0.
        ZeroDiscountOnLifeLoss() if zero_discount_on_life_loss else identity,
        # Select the RGB observation as the main observation, dropping lives.
        select_rgb_observation,
        # obs: 1, 2, 3, 4, 5, 6, 7, 8, 9, ...
        # Write timesteps into a fixed-sized buffer padded with None.
        FixedPaddedBuffer(length=num_action_repeats, initial_index=-1),
        # obs: ~~~1, 2~~~, 23~~, 234~, 2345, 6~~~, 67~~, 678~, 6789, ...
        # Periodically return the deque of timesteps, when the current timestep is
        # FIRST, after that every 4 steps, and when the current timestep is LAST.
        ConditionallySubsample(TimestepBufferCondition(num_action_repeats)),
        # obs: ~~~1, ~, ~, ~, 2345, ~, ~, ~, 6789, ...
        # If None pass through, otherwise apply the processor.
        Maybe(
            Sequential(
                # Replace Nones with zero padding in each buffer.
                none_to_zero_pad,
                # obs: 0001, ~, ~, ~, 2345, ~, ~, ~, 6789, ...
                # Convert sequence of nests into a nest of sequences.
                named_tuple_sequence_stack,
                # Choose representative step type from an array of step types.
                ApplyToNamedTupleField("step_type", reduce_step_type),
                # Rewards: sum then clip.
                ApplyToNamedTupleField(
                    "reward",
                    aggregate_rewards,
                    clip_reward(max_abs_reward) if max_abs_reward else identity,
                ),
                # Discounts: take product and scale by an additional discount.
                ApplyToNamedTupleField(
                    "discount",
                    aggregate_discounts,
                    apply_additional_discount(additional_discount),
                ),
                # Observations: max pool, grayscale, resize, and stack.
                ApplyToNamedTupleField(
                    "observation",
                    lambda obs: np.stack(obs[-num_pooled_frames:], axis=0),
                    lambda obs: np.max(obs, axis=0),
                    # obs: max[01], ~, ~, ~, max[45], ~, ~, ~, max[89], ...
                    # obs:    A,    ~, ~, ~,    B,    ~, ~, ~,    C, ...
                    rgb2y if grayscaling else identity,
                    resize(resize_shape) if resize_shape else identity,
                    Deque(max_length=num_stacked_frames),
                    # obs: A, ~, ~, ~, AB, ~, ~, ~, ABC, ~, ~, ~, ABCD, ~, ~, ~,
                    #      BCDE, ~, ~, ~, CDEF, ...
                    list,
                    trailing_zero_pad(length=num_stacked_frames),
                    # obs: A000, ~, ~, ~, AB00, ~, ~, ~, ABC0, ~, ~, ~, ABCD,
                    #      ~, ~, ~, BCDE, ...
                    lambda obs: np.stack(obs, axis=-1),
                ),
            )
        ),
    )


def processor_pacman(
    additional_discount: float = 0.99,
    max_abs_reward: Optional[float] = 1.0,
    resize_shape: Optional[Tuple[int, int]] = (84, 84),
    num_stacked_frames: int = 4,
    clip_rewards: bool = False,
) -> Processor[[dm_env.TimeStep], Optional[dm_env.TimeStep]]:
    """Standard DQN preprocessing on Atari without framestacking and action repeats"""

    # This processor does the following to a sequence of timesteps.Add framestacking if needed
    # 1. Clips rewards
    # 2. Applies an additional discount.
    # 3. RGB observations even if frame stacking is enabled.
    # 4. Resizes observations.
    # 5. Stacks observations if frame stacking is enabled.

    if clip_rewards:
        return Sequential(
            # Rewards: sum then clip.
            ApplyToNamedTupleField(
                "reward",
                clip_reward(max_abs_reward) if max_abs_reward else identity,
            ),
            # Discounts: take product and scale by an additional discount.
            ApplyToNamedTupleField(
                "discount",
                apply_additional_discount(additional_discount),
            ),
            # Observations: resize
            ApplyToNamedTupleField(
                "observation",
                resize(resize_shape) if resize_shape else identity,
                Deque(max_length=num_stacked_frames),
                # obs: A, ~, ~, ~, AB, ~, ~, ~, ABC, ~, ~, ~, ABCD, ~, ~, ~,
                #      BCDE, ~, ~, ~, CDEF, ...
                list,
                trailing_zero_pad(length=num_stacked_frames),
                # obs: A000, ~, ~, ~, AB00, ~, ~, ~, ABC0, ~, ~, ~, ABCD,
                #      ~, ~, ~, BCDE, ...
                lambda obs: np.stack(obs, axis=-1),
                # reshape to (84, 84, 12) if resize_shape is (84, 84, 3, 4) when using both RGB and frame stacking
                lambda obs: np.reshape(obs, resize_shape + (num_stacked_frames * 3,)) if len(obs.shape) == 4 else obs,
            ),
        )

    else:
        return Sequential(
            # Discounts: take product and scale by an additional discount.
            ApplyToNamedTupleField(
                "discount",
                apply_additional_discount(additional_discount),
            ),
            # Observations: resize
            ApplyToNamedTupleField(
                "observation",
                resize(resize_shape) if resize_shape else identity,
                Deque(max_length=num_stacked_frames),
                # obs: A, ~, ~, ~, AB, ~, ~, ~, ABC, ~, ~, ~, ABCD, ~, ~, ~,
                #      BCDE, ~, ~, ~, CDEF, ...
                list,
                trailing_zero_pad(length=num_stacked_frames),
                # obs: A000, ~, ~, ~, AB00, ~, ~, ~, ABC0, ~, ~, ~, ABCD,
                #      ~, ~, ~, BCDE, ...
                lambda obs: np.stack(obs, axis=-1),
                # reshape to (84, 84, 12) if resize_shape is (84, 84, 3, 4) when using both RGB and frame stacking
                lambda obs: np.reshape(obs, resize_shape + (num_stacked_frames * 3,)) if len(
                    obs.shape) == 4 else obs,
            ),
        )

def processor_minatar(
    additional_discount: float = 0.99,
    max_reward: Optional[float] = 1.0,
    num_stacked_frames: int = 10,
    reward_clipping: bool = True,
) -> Processor[[dm_env.TimeStep], Optional[dm_env.TimeStep]]:

    if reward_clipping:
        return Sequential(
            # Rewards: sum then clip.
            ApplyToNamedTupleField(
                "reward",
                clip_reward(max_reward) if max_reward else identity,
            ),
            # Discounts: take product and scale by an additional discount.
            ApplyToNamedTupleField(
                "discount",
                apply_additional_discount(additional_discount),
            ),
            # Observations: resize
            ApplyToNamedTupleField(
                "observation",
                list,
                zero_pad_to_dimension(axis=-1, max_channels=num_stacked_frames),
            ),
        )

    else:
        return Sequential(
            # Discounts: take product and scale by an additional discount.
            ApplyToNamedTupleField(
                "discount",
                apply_additional_discount(additional_discount),
            ),
            # Observations: resize
            ApplyToNamedTupleField(
                "observation",
                list,
                zero_pad_to_dimension(axis=-1, max_channels=num_stacked_frames),
            ),
        )
