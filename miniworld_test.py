import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import hydra

import pyglet
from pyglet.window import key
import sys
import gym
import numpy as np
from workspace.miniworld_workspace import MiniworldWorkspace


def train(workspace):

    cfg = workspace.config

    train_task = workspace.train_task
    timer = workspace.timer

    head_directions = 4
    num_actions = workspace.action_shape

    # env_vis = train_task.get_env(task_id=0)
    # env_vis = gym.make("MiniWorld-FourRoomsNoLeftTask2-v0")
    env_vis = gym.make("MiniWorld-FourRoomsNoLeftTask2-v0")
    env_vis.reset()

    view_mode = workspace.view_mode
    # window = workspace.window
    # window.reg_key_handler(workspace.key_handler)
    # workspace.start()

    env_vis.render(view=workspace.view_mode)

    print("ok so far!")

    def step(action):
        print(
            "step {}/{}: {}".format(
                env_vis.step_count + 1,
                env_vis.max_episode_steps,
                env_vis.actions(action).name,
            )
        )

        obs, reward, done, info = env_vis.step_with_agent_pos_dir(action)

        print("info: ", info)

        if reward != 0:
            print("reward={:.2f}".format(reward))

        if done:
            print("done!")
            # agent_x = np.random.uniform(-6.5, 6.5)
            # agent_z = np.random.uniform(-6.5, 6.5)
            # env_vis.reset(agent_pos=(agent_x, 0, agent_z))
            env_vis.reset()

        env_vis.render(view=view_mode)

    @env_vis.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        """
        This handler processes keyboard commands that
        control the simulation
        """

        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print("RESET")
            agent_x = np.random.uniform(-6.5, 6.5)
            agent_z = np.random.uniform(-6.5, 6.5)
            env_vis.reset(agent_pos=(agent_x, 0, agent_z))
            # self.env_vis.reset()
            env_vis.render(view=view_mode)
            return

        if symbol == key.ESCAPE:
            env_vis.close()
            sys.exit(0)

        if symbol == key.UP:
            step(env_vis.actions.move_forward)
        elif symbol == key.DOWN:
            step(env_vis.actions.move_back)

        elif symbol == key.LEFT:
            step(env_vis.actions.turn_left)
        elif symbol == key.RIGHT:
            step(env_vis.actions.turn_right)

        elif symbol == key.PAGEUP or symbol == key.P:
            step(env_vis.actions.pickup)
        elif symbol == key.PAGEDOWN or symbol == key.D:
            step(env_vis.actions.drop)

        elif symbol == key.ENTER:
            step(env_vis.actions.done)

    @env_vis.unwrapped.window.event
    def on_key_release(symbol, modifiers):
        pass

    @env_vis.unwrapped.window.event
    def on_draw():
        env_vis.render(view=view_mode)

    @env_vis.unwrapped.window.event
    def on_close():
        pyglet.app.exit()

    pyglet.app.run()

    env_vis.close()


@hydra.main(config_path=".", config_name="miniworld_test", version_base=None)
def main(cfg):

    workspace = MiniworldWorkspace(cfg)
    train(workspace)


if __name__ == "__main__":
    main()
