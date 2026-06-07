"""environment wrapper around miniworld"""
import gym

from typing import Optional, Tuple, Text, Any, Union

import dm_env
import numpy as np
from dm_env import specs
import chex
import math

Array = chex.Array


class GymMiniworld(dm_env.Environment):
    """Gym Miniworld with a `dm_env.Environment` interface."""

    def __init__(self, game: str):
        self._gym_env = gym.make(game, disable_env_checker=True)
        self._start_of_episode = True

    def reset(self) -> dm_env.TimeStep:
        """Resets the environment and starts a new episode."""
        observation, info = self._gym_env.reset()
        timestep = dm_env.restart(observation)
        self._start_of_episode = False
        return timestep

    def reset_with_agent_pos_dir(
        self,
        agent_pos: Tuple,
        agent_dir: int,
    ) -> Tuple[dm_env.TimeStep, bool, int]:
        """
        Reset the environment with a specific agent position and direction.

        Parameters
        ----------
        agent_pos: Tuple - (x, y) position of the agent
        agent_dir: int - head direction of the agent in degrees

        Returns
        -------
        timestep: dm_env.TimeStep
        is_valid_position: bool - whether the agent position is valid or not
        room_idx: int - index of the room in which the agent is located
        """

        obs, is_valid_position, room_idx = self._gym_env.reset_with_agent_pos_dir(
            agent_pos=agent_pos, agent_dir=agent_dir
        )

        timestep = dm_env.restart(obs)
        self._start_of_episode = False
        return timestep, is_valid_position, room_idx

    def reset_with_agent_in_room(
        self,
        agent_dir: int,
        room_idx: int,
    ) -> Tuple[dm_env.TimeStep, tuple, bool, int]:
        """
        Reset the environment with a specific agent position and direction.

        Parameters
        ----------
        agent_dir
        room_idx

        Returns
        -------
        timestep: dm_env.TimeStep
        is_valid_position: bool - whether the agent position is valid or not
        room_idx: int - index of the room in which the agent is located
        """

        (
            obs,
            agent_pos,
            is_valid_position,
            room_idx,
        ) = self._gym_env.reset_with_agent_in_room(
            agent_dir=agent_dir, room_idx=room_idx
        )

        # convert agent_pos to two-decimal float
        agent_pos = tuple([round(x, 2) for x in agent_pos])

        timestep = dm_env.restart(obs)
        self._start_of_episode = False
        return timestep, agent_pos, is_valid_position, room_idx

    def reset_with_state_info(self) -> Tuple[dm_env.TimeStep, dict[Text, Any]]:
        """
        Reset the environment and return the agent position and direction.
        Returns
        -------
        timestep: dm_env.TimeStep
        info: dict which contains the keys "agent_pos" and "agent_dir"

        """
        observation, info = self._gym_env.reset()
        timestep = dm_env.restart(observation)

        # convert agent_dir in info from radians to degrees
        info["agent_dir"] = int(info["agent_dir"] * 180 / math.pi) % 360
        return timestep, info

    def step(self, action: int) -> dm_env.TimeStep:
        """Updates the environment according to the action."""
        observation, reward, done, info = self._gym_env.step(action)
        if done:
            timestep = dm_env.termination(reward, observation)
        else:
            timestep = dm_env.transition(reward, observation)
        return timestep

    def step_with_agent_pos_dir(self, action: np.int32) -> dict[str, Any]:
        """
        To be used for debugging and visualization purposes only.
        Parameters
        ----------
        action: np.int32

        Returns
        -------
        output: dict which contains the keys "observation", "reward", "done", "truncated", "agent_pos", "agent_dir"

        """
        observation, reward, done, info = self._gym_env.step_with_agent_pos_dir(action)
        output = dict()
        output["reward"] = reward
        output["done"] = done
        for key in info.keys():
            output[key] = info[key]

        # convert agent_pos to two-decimal float
        output["agent_pos"] = tuple([round(x, 2) for x in output["agent_pos"]])

        # replace agent_pos[1] with agent_pos[2] and remove agent_pos[2] from output["agent_pos"]
        output["agent_pos"] = (output["agent_pos"][0], output["agent_pos"][2])
        obs_with_info = {"observation": observation, "info": output}
        if done:
            timestep = dm_env.termination(reward, obs_with_info)
        else:
            timestep = dm_env.transition(reward, obs_with_info)
        return timestep

    def observation_spec(self) -> specs.Array:
        """Returns the observation spec."""
        return specs.Array(
            shape=self._gym_env.observation_space.shape, dtype=np.float32
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(self._gym_env.action_space.n, name="action")

    def render(self, view: str = Union["agent", "top"]):
        """Renders the environment."""
        self._gym_env.render("pyglet", view=view)

    def unwrapped(self):
        return self._gym_env

    def close(self):
        self._gym_env.close()

    def window(self):
        return self._gym_env.window

    def min_x(self) -> float:
        """
        Return the minimum x coordinate of the floor plan.

        Returns
        -------
        min_x: float
        """
        return self._gym_env.min_x

    def max_x(self) -> float:
        """
        Return the maximum x coordinate of the floor plan.

        Returns
        -------
        max_x: float
        """
        return self._gym_env.max_x

    def min_z(self) -> float:
        """
        Return the minimum z coordinate of the floor plan.

        Returns
        -------
        min_z: float
        """
        return self._gym_env.min_z

    def max_z(self) -> float:
        """
        Return the maximum z coordinate of the floor plan.

        Returns
        -------
        max_z: float

        """
        return self._gym_env.max_z

    @staticmethod
    def convert_attributes_to_mid_step_timestep(
        observation: Array, reward: float
    ) -> dm_env.TimeStep:
        """
        Convert attributes to a mid-step timestep.
        To be used for debugging and visualization purposes only.
        Parameters
        ----------
        observation
        reward

        Returns
        -------
        timestep: dm_env.TimeStep with step_type=dm_env.StepType.MID and discount=1.0

        """
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            observation=observation,
            reward=reward,
            discount=1.0,
        )
        return timestep

    @staticmethod
    def convert_attributes_to_last_step_timestep(
        observation: Array, reward: float
    ) -> dm_env.TimeStep:
        """
        Convert attributes to a last-step timestep.
        To be used for debugging and visualization purposes only.
        Parameters
        ----------
        observation
        reward

        Returns
        -------
        timestep: dm_env.TimeStep with step_type=dm_env.StepType.LAST and discount=0.0

        """
        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.LAST,
            observation=observation,
            reward=reward,
            discount=0.0,
        )
        return timestep
