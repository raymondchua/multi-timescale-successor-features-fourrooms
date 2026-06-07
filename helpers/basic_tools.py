import os
import jax
import pickle
import random
import numpy as np
import time
from collections.abc import MutableMapping


class LinearSchedule:
    """Linear schedule, used for exploration epsilon in DQN agents."""

    def __init__(self, begin_value, end_value, begin_t, end_t=None, decay_steps=None):
        if (end_t is None) == (decay_steps is None):
            raise ValueError("Exactly one of end_t, decay_steps must be provided.")
        self._decay_steps = decay_steps if end_t is None else end_t - begin_t
        self._begin_t = begin_t
        self._begin_value = begin_value
        self._end_value = end_value

    def __call__(self, t):
        """Implements a linear transition from a begin to an end value."""
        frac = min(max(t - self._begin_t, 0), self._decay_steps) / self._decay_steps
        return (1 - frac) * self._begin_value + frac * self._end_value


class NullCheckpoint:
    """A placeholder checkpointing object that does nothing.

    Can be used as a substitute for an actual checkpointing object when
    checkpointing is disabled.
    """

    def __init__(self):
        self.state = AttributeDict()

    def save(self) -> None:
        pass

    def can_be_restored(self) -> bool:
        return False

    def restore(self) -> None:
        pass


class Checkpointer:
    def __init__(self, path):
        self.path = path

    def save(self, params, epoch, args):
        params = jax.device_get(params)
        with open(
            os.path.join(
                self.path,
                "{agent}_seed{seed}_task{task}_epoch{epoch}_tube{tube}.pkl".format(
                    agent=args.agent,
                    seed=args.seed,
                    task=args.task_id,
                    epoch=epoch,
                    tube=args.init_tube,
                ),
            ),
            "wb",
        ) as fp:
            pickle.dump(params, fp)

    def load(self, agent, seed, task_id, epoch, init_tube):
        with open(
            os.path.join(
                self.path,
                "{agent}_seed{seed}_task{task}_epoch{epoch}_tube{tube}.pkl".format(
                    agent=agent, seed=seed, task=task_id, epoch=epoch, tube=init_tube
                ),
            ),
            "rb",
        ) as fp:
            params = pickle.load(fp)
        print("Weights load success!")
        return jax.device_put(params)

    def load_wo_tube(self, agent, seed, task_id, epoch):
        with open(
            os.path.join(
                self.path,
                "{agent}_seed{seed}_task{task}_epoch{epoch}.pkl".format(
                    agent=agent, seed=seed, task=task_id, epoch=epoch
                ),
            ),
            "rb",
        ) as fp:
            params = pickle.load(fp)
        print("Weights load success!")
        return jax.device_put(params)


class AttributeDict(dict):
    """A `dict` that supports getting, setting, deleting keys via attributes."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def dictionary_flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(dictionary_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Timer:
    """
    Source: https://github.com/rll-research/url_benchmark/utils.py
    """

    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class Until:
    """
    Source: https://github.com/rll-research/url_benchmark/utils.py
    """

    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until
