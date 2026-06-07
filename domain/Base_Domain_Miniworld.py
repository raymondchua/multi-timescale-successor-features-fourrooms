from domain.Base_Domain import BaseDomain, BaseDomainKwargs
from typing_extensions import Unpack
import numpy as np

from helpers.dm_env_wrappers.dm_env_miniworld import GymMiniworld
from helpers import processor_miniworld

import gym_miniworld  # This registers the environments
import gym_miniworld.envs


class BaseDomainMiniworldKwargs(BaseDomainKwargs):
    clip_reward: bool
    eval_env_diff_train_env: bool
    top_down_view: bool
    max_abs_reward: float
    max_episode_length: int
    short_name: str
    slippery_prob: float
    visualise_rep: bool
    visualise_rep_before_training: bool

class BaseDomainMiniworld(BaseDomain):
    def __init__(self, **kwargs: Unpack[BaseDomainMiniworldKwargs]):
        super().__init__(**kwargs)
        self._kwargs = kwargs
        self._actions_name = ["Turn left", "Turn right", "Forward", "Move back"]
        self._env = None
        self._env1 = None
        self._env2 = None
        self._envs = []
        self._envs_short_names = []
        self._env_type = "miniworld"

        self._eval_env1 = None
        self._eval_env2 = None
        self._eval_env = None

        print("In BaseDomainMiniworld")

    def build_env(self, env: str):
        env = GymMiniworld(
            env,
        )
        return env

    def get_env(self, task_id: int, exposure_id: int = 0):
        # free memory on self._env
        if self._env is not None:
            del self._env

        if task_id % 2 == 0:
            self._env = self._env1

        else:
            self._env = self._env2
        self._env.reset()
        return self._env

    def get_random_env(self):
        rand_index = np.random.randint(low=0, high=len(self._envs))
        env = self._envs[rand_index]
        env.reset()
        return env

    def action_spec(self):
        return self._env1.action_spec()

    @property
    def action_repeat(self):
        return self._kwargs["action_repeat"]

    def observation_spec(self):
        return self._env1.observation_spec()

    def reset(self, task_id):
        self.get_env(task_id).reset()

    @property
    def action_names(self):
        return self._actions_name

    def preprocessor(self):
        return processor_miniworld(
            additional_discount=self._kwargs["discount"],
            max_reward=self._kwargs["max_abs_reward"],
            resize_shape=(
                self._kwargs["environment_height"],
                self._kwargs["environment_width"],
            ),
            use_framestack=self._kwargs["use_framestack"],
            num_stacked_frames=self._kwargs["num_stacked_frames"],
        )

    @property
    def min_returns(self) -> float:
        return self._kwargs["min_returns"]

    @property
    def max_returns(self) -> float:
        return self._kwargs["max_returns"]

    @property
    def eval_env_diff_train_env(self) -> bool:
        return self._kwargs["eval_env_diff_train_env"]

    def get_eval_env(self, task_id: int, exposure_id: int = 0):
        if self._kwargs["eval_env_diff_train_env"]:
            # free memory on self._env
            if self._eval_env is not None:
                del self._eval_env

            if task_id % 2 == 0:
                self._eval_env = self._eval_env1

            else:
                self._eval_env = self._eval_env2
            self._eval_env.reset()
            return self._eval_env

        else:
            return self.get_env(task_id)
