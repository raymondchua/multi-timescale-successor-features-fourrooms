import gym

from .workspace import BaseWorkspace

# from pyglet.window import Window

class MiniworldWorkspace(BaseWorkspace):
    """
    Workspace for MiniWorld
    """

    def __init__(self, cfg, top_view: bool = False, **kwargs):
        super().__init__(cfg, **kwargs)

        self._max_steps_per_episode = cfg.domain.max_episode_length
        self._prev_sf = None
        self._prev_q_rep = None
        self._prev_agent_pos_x = None
        self._prev_agent_pos_y = None
        self._env_vis = None

        self._view_mode = "top" if cfg.domain.visualization.top_view else "agent"
        self._window = None

    @property
    def max_steps_per_episode(self):
        return self._max_steps_per_episode

    @property
    def env_vis(self) -> gym.Env:
        assert self._env_vis is not None
        return self._env_vis

    @property
    def view_mode(self):
        return self._view_mode

    @env_vis.setter
    def env_vis(self, env):
        self._env_vis = env
        self._window = env.window
        self._env_vis.reset()

