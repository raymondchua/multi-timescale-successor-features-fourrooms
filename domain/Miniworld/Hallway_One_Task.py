from typing_extensions import Unpack
from domain.Base_Domain_Miniworld import (
    BaseDomainMiniworld,
    BaseDomainMiniworldKwargs,
)

class HallwayOneTask(BaseDomainMiniworld):
    """
    his domain is a 3D Hallway environment the goal at the end of the hallway.
    The agent starts randomly in the hallway.
    The agent receives a reward of 1 when it reaches goal.
    The agent always see the first person view of the environment.
    """

    def __init__(self, **kwargs: Unpack[BaseDomainMiniworldKwargs]):
        super().__init__(**kwargs)
        print("building domain 49")
        self._kwargs = kwargs

        self._env1 = self.build_env("MiniWorld-Hallway-v0")

        self.observation_space = self._env1.observation_spec().shape
        self._envs = [self._env1]
        self._envs_short_names = ["D49_T1"]

        self._main_rooms_id = [0]
        self._all_rooms_id = [0]

        print("env build")

    @property
    def all_rooms_id(self):
        return self._all_rooms_id

    @property
    def main_rooms_id(self):
        return self._main_rooms_id

    @property
    def connecting_rooms_id(self):
        return None