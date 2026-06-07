from typing import Any, Iterable, Mapping, Optional, Text, Tuple, Callable
import jax
import jax.numpy as jnp
import dm_env
import numpy as np

from helpers import basic_tools, CoverageLoggerCSV, modulators
from collections import defaultdict

Action = int

def run_loop_slippery(
    agent,
    environment: dm_env.Environment,
    discount: float,
    meta_update_step: int,
    global_step_train: int,
    learn_period: int,
    max_steps_per_episode: int,
    meta: Optional[jnp.ndarray],
    num_seed_frames: int,
    num_train_frames: int,
    steps_done_in_current_task: int,
    task_id: int,
    train_agent_action_shape: int,
    slippery_modulator: modulators.NoisySineModulator,
    yield_before_reset: bool = False,
    env_type: Optional[str] = None,
    save_coverage_frequency: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Iterable[
    Tuple[
        dm_env.Environment,
        Optional[dm_env.TimeStep],
        Any,
        Optional[Action],
        Mapping[Text, Any],
    ]
]:
    """Repeatedly alternates step calls on environment and agent.
    At time `t`, `t + 1` environment timesteps and `t + 1` agent steps have been
    seen in the current episode. `t` resets to `0` for the next episode.
    Args:
        agent: Agent to be run, has methods `step(timestep)` and `reset()`.
        environment: Environment to run, has methods `step(action)` and `reset()`.
        discount: Discount factor
        episode, the episode is truncated.
        meta_update_step: How often to update the task
        global_step_train: Amount of steps taken so far for the one epoch
        learn_period: How often to learn from the minibatch (this is learning the rest of the network except the task)
        max_steps_per_episode: If positive, when time t reaches this value within an episode, the episode is truncated.
        meta: task for the agent
        num_seed_frames: Number of frames for the agent to interact with the environment and storing the experiences in
                        the replay buffer, before agent starts to learn
        num_train_frames: Number of steps for one epoch
        steps_done_in_current_task: Number steps taken for current task
        task_id: The id of the current task
        yield_before_reset: Whether to additionally yield `(environment, None,
        agent, None)` before the agent and environment is reset at the start of
        each episode.

    Yields:
      Tuple `(environment, timestep_t, agent, a_t, metrics)` where
      `a_t = agent.step(timestep_t)`.
      and metrics contain information on meta_update_step
      and train_meta_difference.
    """
    global_episode = 0
    episode_step = 0
    episode_reward = 0
    epoch_step = 0

    # num_train_frames and num_seed_frames already take into account the action repeat
    train_until_step = basic_tools.Until(num_train_frames)
    seed_until_step = basic_tools.Until(num_seed_frames)

    agent.reset()
    metrics = dict()

    metrics["meta_update_step"] = meta_update_step
    metrics["train_meta_difference"] = 0.0

    if hasattr(agent, "init_meta"):
        assert meta is not None
    else:
        assert meta is None

    if hasattr(agent, "update_task_every_step"):
        every = agent.update_task_every_step
    else:
        every = 1
    previous_meta = meta

    logger = CoverageLoggerCSV(output_dir=output_dir)
    timestep_t = environment.reset()

    while train_until_step(epoch_step):

        if yield_before_reset:
            yield environment, None, agent, None, metrics

        while True:  # For each step in the current episode.

            if hasattr(agent, "update_task_every_step"):

                if (
                    not seed_until_step(steps_done_in_current_task)
                    and steps_done_in_current_task % every == 0
                ):
                    learn_meta = True

                else:
                    learn_meta = False
            else:
                learn_meta = False

            if (
                not seed_until_step(steps_done_in_current_task)
                and steps_done_in_current_task % learn_period == 0
            ):
                time_to_learn = True

            else:
                time_to_learn = False

            # If it is an Atari environment, timestep_t should have observation that contains
            # pixels, lives, terminated, truncated, info
            step_metrics = agent.step(
                timestep=timestep_t,
                time_to_learn=time_to_learn,
                learn_meta=learn_meta,
                task=meta,
                task_id=task_id,
            )

            a_t = step_metrics["action"]
            slip_prob_threshold = slippery_modulator.sample(global_step_train)
            if np.random.uniform() < slip_prob_threshold:
                while True:
                    new_rand_action = np.random.randint(0, train_agent_action_shape)
                    if new_rand_action != a_t:
                        a_t = new_rand_action
                        break

            step_metrics["slip_prob_threshold"] = slip_prob_threshold
            # update meta if meta is in step_metrics
            if "meta" in step_metrics:
                meta = step_metrics["meta"]
                metrics["meta_update_step"] += 1

                metrics["train_meta_difference"] = jax.device_get(
                    jnp.linalg.norm(previous_meta - meta, ord=2)
                ).item()
                del step_metrics["meta"]

                previous_meta = meta

            metrics.update(step_metrics)
            metrics["steps_done_in_current_task"] = steps_done_in_current_task

            if "action" in metrics:
                del metrics["action"]

            # there is a bug here. We should only yield if timestep_t is not last. However, since we have used this
            # function so many times, we will leave it as it is for now. The correct way to do it is to check if
            # timestep_t is not last.
            # if not timestep_t.last():
            #     yield environment, timestep_t, agent, a_t, metrics
            yield environment, timestep_t, agent, a_t, metrics

            # Update t after one environment step and agent step and relabel.
            episode_step += 1
            a_tm1 = a_t
            if env_type == "miniworld" and save_coverage_frequency is not None:
                timestep_t = environment.step_with_agent_pos_dir(a_tm1)
                # log the agent x and y position and direction
                info = timestep_t.observation["info"]
                logger.log(info["agent_pos"][0], info["agent_pos"][1], info["agent_dir"], timestep_t.reward)

                if global_step_train % save_coverage_frequency == 0:
                    logger.save(global_step_train)

                # remove info from the observation as it is no longer needed and replace observation
                del timestep_t.observation["info"]
                timestep_t = timestep_t._replace(observation=timestep_t.observation["observation"])

            else:
                timestep_t = environment.step(a_tm1)

            episode_reward += timestep_t.reward
            agent.add_to_g(reward=timestep_t.reward)

            global_step_train += 1
            epoch_step += 1
            steps_done_in_current_task += 1

            # max steps per episode minus 1 as we still take one step after the episode is done
            # so that the timestep can be stored in the replay buffer for learning
            # but the action will not be sent to the environment
            if episode_step == max_steps_per_episode - 1:
                timestep_t = timestep_t._replace(step_type=dm_env.StepType.LAST)

            if (
                not seed_until_step(steps_done_in_current_task)
                and steps_done_in_current_task % learn_period == 0
            ):
                time_to_learn_last_timestep = True
            else:
                time_to_learn_last_timestep = False

            if timestep_t.last():
                agent.update_td_scaler(reward=timestep_t.reward, discount=0, returns=agent.g)
                agent.reset_G()
                unused_action_and_metrics = agent.step(
                    timestep=timestep_t,
                    time_to_learn=time_to_learn_last_timestep,
                    task=meta,
                    task_id=task_id,
                )

                metrics.update(unused_action_and_metrics)

                if "action" in metrics:
                    del metrics["action"]

                yield environment, timestep_t, agent, None, metrics
                global_episode += 1
                timestep_t = environment.reset()  # timestep_0.
                if hasattr(agent, "init_meta"):
                    meta = agent.init_meta()
                episode_step = 0
                episode_reward = 0
                agent.reset()  # reset the agent at the end of the episode

                break

            # update td scaler
            agent.update_td_scaler(reward=timestep_t.reward, discount=discount, returns=None)


def run_loop_slippery_non_periodic(
    agent,
    environment: dm_env.Environment,
    discount: float,
    meta_update_step: int,
    global_step_train: int,
    learn_period: int,
    max_steps_per_episode: int,
    meta: Optional[jnp.ndarray],
    num_seed_frames: int,
    num_train_frames: int,
    steps_done_in_current_task: int,
    task_id: int,
    train_agent_action_shape: int,
    slippery_modulator: modulators.NoisyAPeriodicSineModulator | modulators.OUDrift,
    yield_before_reset: bool = False,
    env_type: Optional[str] = None,
    save_coverage_frequency: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Iterable[
    Tuple[
        dm_env.Environment,
        Optional[dm_env.TimeStep],
        Any,
        Optional[Action],
        Mapping[Text, Any],
    ]
]:
    """Repeatedly alternates step calls on environment and agent.
    At time `t`, `t + 1` environment timesteps and `t + 1` agent steps have been
    seen in the current episode. `t` resets to `0` for the next episode.
    Args:
        agent: Agent to be run, has methods `step(timestep)` and `reset()`.
        environment: Environment to run, has methods `step(action)` and `reset()`.
        discount: Discount factor
        episode, the episode is truncated.
        meta_update_step: How often to update the task
        global_step_train: Amount of steps taken so far for the one epoch
        learn_period: How often to learn from the minibatch (this is learning the rest of the network except the task)
        max_steps_per_episode: If positive, when time t reaches this value within an episode, the episode is truncated.
        meta: task for the agent
        num_seed_frames: Number of frames for the agent to interact with the environment and storing the experiences in
                        the replay buffer, before agent starts to learn
        num_train_frames: Number of steps for one epoch
        steps_done_in_current_task: Number steps taken for current task
        task_id: The id of the current task
        yield_before_reset: Whether to additionally yield `(environment, None,
        agent, None)` before the agent and environment is reset at the start of
        each episode.

    Yields:
      Tuple `(environment, timestep_t, agent, a_t, metrics)` where
      `a_t = agent.step(timestep_t)`.
      and metrics contain information on meta_update_step
      and train_meta_difference.
    """
    global_episode = 0
    episode_step = 0
    episode_reward = 0
    epoch_step = 0

    # num_train_frames and num_seed_frames already take into account the action repeat
    train_until_step = basic_tools.Until(num_train_frames)
    seed_until_step = basic_tools.Until(num_seed_frames)

    agent.reset()
    metrics = dict()

    metrics["meta_update_step"] = meta_update_step
    metrics["train_meta_difference"] = 0.0

    if hasattr(agent, "init_meta"):
        assert meta is not None
    else:
        assert meta is None

    if hasattr(agent, "update_task_every_step"):
        every = agent.update_task_every_step
    else:
        every = 1
    previous_meta = meta

    logger = CoverageLoggerCSV(output_dir=output_dir)
    timestep_t = environment.reset()

    while train_until_step(epoch_step):

        if yield_before_reset:
            yield environment, None, agent, None, metrics

        while True:  # For each step in the current episode.

            if hasattr(agent, "update_task_every_step"):

                if (
                    not seed_until_step(steps_done_in_current_task)
                    and steps_done_in_current_task % every == 0
                ):
                    learn_meta = True

                else:
                    learn_meta = False
            else:
                learn_meta = False

            if (
                not seed_until_step(steps_done_in_current_task)
                and steps_done_in_current_task % learn_period == 0
            ):
                time_to_learn = True

            else:
                time_to_learn = False

            # If it is an Atari environment, timestep_t should have observation that contains
            # pixels, lives, terminated, truncated, info
            step_metrics = agent.step(
                timestep=timestep_t,
                time_to_learn=time_to_learn,
                learn_meta=learn_meta,
                task=meta,
                task_id=task_id,
            )

            a_t = step_metrics["action"]
            slip_prob_threshold = slippery_modulator.sample()
            if np.random.uniform() < slip_prob_threshold:
                while True:
                    new_rand_action = np.random.randint(0, train_agent_action_shape)
                    if new_rand_action != a_t:
                        a_t = new_rand_action
                        break

            step_metrics["slip_prob_threshold"] = slip_prob_threshold
            # update meta if meta is in step_metrics
            if "meta" in step_metrics:
                meta = step_metrics["meta"]
                metrics["meta_update_step"] += 1

                metrics["train_meta_difference"] = jax.device_get(
                    jnp.linalg.norm(previous_meta - meta, ord=2)
                ).item()
                del step_metrics["meta"]

                previous_meta = meta

            metrics.update(step_metrics)
            metrics["steps_done_in_current_task"] = steps_done_in_current_task

            if "action" in metrics:
                del metrics["action"]

            # there is a bug here. We should only yield if timestep_t is not last. However, since we have used this
            # function so many times, we will leave it as it is for now. The correct way to do it is to check if
            # timestep_t is not last.
            # if not timestep_t.last():
            #     yield environment, timestep_t, agent, a_t, metrics
            yield environment, timestep_t, agent, a_t, metrics

            # Update t after one environment step and agent step and relabel.
            episode_step += 1
            a_tm1 = a_t
            if env_type == "miniworld" and save_coverage_frequency is not None:
                timestep_t = environment.step_with_agent_pos_dir(a_tm1)
                # log the agent x and y position and direction
                info = timestep_t.observation["info"]
                logger.log(info["agent_pos"][0], info["agent_pos"][1], info["agent_dir"], timestep_t.reward)

                if global_step_train % save_coverage_frequency == 0:
                    logger.save(global_step_train)

                # remove info from the observation as it is no longer needed and replace observation
                del timestep_t.observation["info"]
                timestep_t = timestep_t._replace(observation=timestep_t.observation["observation"])

            else:
                timestep_t = environment.step(a_tm1)

            episode_reward += timestep_t.reward
            agent.add_to_g(reward=timestep_t.reward)

            global_step_train += 1
            epoch_step += 1
            steps_done_in_current_task += 1

            # max steps per episode minus 1 as we still take one step after the episode is done
            # so that the timestep can be stored in the replay buffer for learning
            # but the action will not be sent to the environment
            if episode_step == max_steps_per_episode - 1:
                timestep_t = timestep_t._replace(step_type=dm_env.StepType.LAST)

            if (
                not seed_until_step(steps_done_in_current_task)
                and steps_done_in_current_task % learn_period == 0
            ):
                time_to_learn_last_timestep = True
            else:
                time_to_learn_last_timestep = False

            if timestep_t.last():
                agent.update_td_scaler(reward=timestep_t.reward, discount=0, returns=agent.g)
                agent.reset_G()
                unused_action_and_metrics = agent.step(
                    timestep=timestep_t,
                    time_to_learn=time_to_learn_last_timestep,
                    task=meta,
                    task_id=task_id,
                )

                metrics.update(unused_action_and_metrics)

                if "action" in metrics:
                    del metrics["action"]

                yield environment, timestep_t, agent, None, metrics
                global_episode += 1
                timestep_t = environment.reset()  # timestep_0.
                if hasattr(agent, "init_meta"):
                    meta = agent.init_meta()
                episode_step = 0
                episode_reward = 0
                agent.reset()  # reset the agent at the end of the episode

                break

            # update td scaler
            agent.update_td_scaler(reward=timestep_t.reward, discount=discount, returns=None)