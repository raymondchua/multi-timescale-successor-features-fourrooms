from typing_extensions import Unpack
from domain.Base_Domain_Miniworld import (
    BaseDomainMiniworld,
    BaseDomainMiniworldKwargs,
)

class MiniworldTwoTasks(BaseDomainMiniworld):
    """
    This domain is a 3D Four Rooms environment with two goals. One in the top left room and the other in the bottom
    right room. The agent starts randomly in one of the rooms.
    The agent receives a reward of 1 when it reaches the correct goal and a reward of -1 when it reaches the wrong goal.
    The agent always see the first person view of the environment.
    """

    def __init__(self, **kwargs: Unpack[BaseDomainMiniworldKwargs]):
        super().__init__(**kwargs)
        print("building domain 20")
        self._kwargs = kwargs

        self._env1 = self.build_env("MiniWorld-FourRoomsTask1-v0")
        self._env2 = self.build_env("MiniWorld-FourRoomsTask2-v0")

        self.observation_space = self._env1.observation_spec().shape
        self._envs = [self._env1, self._env2]
        self._envs_short_names = ["D20_T1", "D20_T2"]

        self._main_rooms_id = [0, 1, 2, 3]
        self._connecting_rooms_id = [4, 5, 6, 7]
        self._all_rooms_id = [0, 1, 2, 3, 4, 5, 6, 7]

        print("env build")

    @property
    def all_rooms_id(self):
        return self._all_rooms_id

    @property
    def main_rooms_id(self):
        return self._main_rooms_id

    @property
    def connecting_rooms_id(self):
        return self._connecting_rooms_id
