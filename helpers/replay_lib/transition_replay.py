import collections
import typing
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Text,
    Tuple,
)

import numpy as np
import snappy

from .replay_helpers import (
    importance_sampling_weights,
    PrioritizedDistribution,
    ReplayStructure,
)


class TransitionReplay(Generic[ReplayStructure]):
    """Uniform replay, with circular buffer storage for flat named tuples."""

    def __init__(
        self,
        capacity: int,
        structure: ReplayStructure,
        random_state: np.random.RandomState,
        encoder: Optional[Callable[[ReplayStructure], Any]] = None,
        decoder: Optional[Callable[[Any], ReplayStructure]] = None,
    ):
        self._capacity = capacity
        self._structure = structure
        self._random_state = random_state
        self._encoder = encoder or (lambda s: s)
        self._decoder = decoder or (lambda s: s)

        self._storage = [None] * capacity
        self._num_added = 0

    def add(self, item: ReplayStructure) -> None:
        """Adds single item to replay."""
        # this part may need to be fix to account for num_envs
        self._storage[self._num_added % self._capacity] = self._encoder(item)
        self._num_added += 1

    def get(self, indices: Sequence[int]) -> List[ReplayStructure]:
        """Retrieves items by indices."""
        return [self._decoder(self._storage[i]) for i in indices]

    def sample(self, size: int) -> ReplayStructure:
        """Samples batch of items from replay uniformly, with replacement."""
        indices = self._random_state.randint(self.size, size=size)
        samples = self.get(indices)
        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]
        return type(self._structure)(*stacked)  # pytype: disable=not-callable

    def uniform_sample(self, size: int) -> ReplayStructure:
        # For normal replay buffer, uniform sample is the same as sample
        return self.sample(size)

    @property
    def size(self) -> int:
        """Number of items currently contained in replay."""
        return min(self._num_added, self._capacity)

    @property
    def capacity(self) -> int:
        """Total capacity of replay (max number of items stored at any one time)."""
        return self._capacity

    def get_state(self) -> Mapping[Text, Any]:
        """Retrieves replay state as a dictionary (e.g. for serialization)."""
        return {
            "storage": self._storage,
            "num_added": self._num_added,
        }

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets replay state from a (potentially de-serialized) dictionary."""
        self._storage = state["storage"]
        self._num_added = state["num_added"]

    def reset(self) -> None:
        """Empty the replay buffer"""
        self._storage = [None] * self._capacity
        self._num_added = 0

class PrioritizedTransitionReplay(Generic[ReplayStructure]):
    """Prioritized replay, with circular buffer storage for flat named tuples.
    This is the proportional variant as described in
    http://arxiv.org/abs/1511.05952.
    """

    def __init__(
        self,
        capacity: int,
        structure: ReplayStructure,
        priority_exponent: float,
        importance_sampling_exponent: Callable[[int], float],
        uniform_sample_probability: float,
        normalize_weights: bool,
        random_state: np.random.RandomState,
        encoder: Optional[Callable[[ReplayStructure], Any]] = None,
        decoder: Optional[Callable[[Any], ReplayStructure]] = None,
    ):
        self._capacity = capacity
        self._structure = structure
        self._priority_exponent = priority_exponent
        self._uniform_sample_probability = uniform_sample_probability
        self._random_state = random_state
        self._encoder = encoder or (lambda s: s)
        self._decoder = decoder or (lambda s: s)
        self._distribution = PrioritizedDistribution(
            capacity=self._capacity,
            priority_exponent=self._priority_exponent,
            uniform_sample_probability=self._uniform_sample_probability,
            random_state=self._random_state,
        )
        self._importance_sampling_exponent = importance_sampling_exponent
        self._normalize_weights = normalize_weights
        self._storage = [None] * capacity
        self._t = 0

    def add(self, item: ReplayStructure, priority: float) -> None:
        """Adds a single item with a given priority to the replay buffer."""
        index = self._t % self._capacity
        self._distribution.set_priorities([index], [priority])
        self._storage[index] = self._encoder(item)
        self._t += 1

    def get(self, indices: Sequence[int]) -> List[ReplayStructure]:
        """Retrieves transitions by indices."""
        return [self._decoder(self._storage[i]) for i in indices]

    def sample(
        self,
        size: int,
    ) -> Tuple[ReplayStructure, np.ndarray, np.ndarray]:
        """Samples a batch of transitions."""
        indices, probabilities = self._distribution.sample(size)
        weights = importance_sampling_weights(
            probabilities,
            uniform_probability=1.0 / self.size,
            exponent=self.importance_sampling_exponent,
            normalize=self._normalize_weights,
        )
        samples = self.get(indices)
        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]
        # pytype: disable=not-callable
        return type(self._structure)(*stacked), indices, weights
        # pytype: enable=not-callable

    def uniform_sample(self, size: int) -> ReplayStructure:
        """Samples batch of items from replay uniformly, with replacement."""
        indices = self._random_state.randint(self.size, size=size)
        samples = self.get(indices)
        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]
        return type(self._structure)(*stacked)  # pytype: disable=not-callable

    def update_priorities(
        self, indices: Sequence[int], priorities: Sequence[float]
    ) -> None:
        """Updates indices with given priorities."""
        priorities = np.asarray(priorities)
        self._distribution.update_priorities(indices, priorities)

    @property
    def size(self) -> int:
        """Number of elements currently contained in replay."""
        return min(self._t, self._capacity)

    @property
    def capacity(self) -> int:
        """Total capacity of replay (maximum number of items that can be stored)."""
        return self._capacity

    @property
    def importance_sampling_exponent(self):
        """Importance sampling exponent at current step."""
        return self._importance_sampling_exponent(self._t)

    def get_state(self) -> Mapping[Text, Any]:
        """Retrieves replay state as a dictionary (e.g. for serialization)."""
        return {
            "t": self._t,
            "storage": self._storage,
            "distribution": self._distribution.get_state(),
        }

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets replay state from a (potentially de-serialized) dictionary."""
        self._t = state["t"]
        self._storage = state["storage"]
        self._distribution.set_state(state["distribution"])

    def reset(self) -> None:
        """Empty the replay buffer"""
        self._storage = [None] * self._capacity
        self._t = 0
        self._distribution = PrioritizedDistribution(
            capacity=self._capacity,
            priority_exponent=self._priority_exponent,
            uniform_sample_probability=self._uniform_sample_probability,
            random_state=self._random_state,
        )
