from typing import Any, Iterable, Mapping, Optional, Text, Tuple
import jax
import jax.numpy as jnp
import dm_env
import numpy as np

from helpers import basic_tools, modulators

Action = int

def run_loop_slippery(
    eval_agent,
    environment: dm_env.Environment,
    train_agent_action_shape: int,
    max_steps_per_episode: int,
    num_eval_frames: int,
    global_step_train: int,
    slippery_modulator: modulators.NoisySineModulator,
    yield_before_reset: bool = False,
    meta: Optional[jnp.ndarray] = None,
    env_type: Optional[str] = None,
    mask: Optional[jnp.ndarray] = None,
    recall_gain: Optional[jnp.ndarray] = None,
) -> Iterable[
    Tuple[
        dm_env.Environment,
        Optional[dm_env.TimeStep],
        Any,
        Optional[Action],
        Mapping[Text, Any],
    ]
]:
    """The standard run loop function.
    At time `t`, `t + 1` environment timesteps and `t + 1` agent steps have been
    seen in the current episode. `t` resets to `0` for the next episode.
    Args:
        eval_agent: Agent to be run, has methods `step(timestep)` and `reset()`.
        environment: Environment to run, has methods `step(action)` and `reset()`.
        episode, the episode is truncated.
        max_steps_per_episode: If positive, when time t reaches this value within an episode, the episode is truncated.
        num_eval_frames: Number of frames for evaluation
        yield_before_reset: Whether to additionally yield `(environment, None,
        agent, None)` before the agent and environment is reset at the start of
        each episode.
        meta: Optional task for the agent if it is APS variant

    Yields:
      Tuple `(environment, timestep_t, agent, a_t, metrics)` where
      `a_t = agent.step(timestep_t)`.
      and metrics contain information on state_frequency, state_action_frequency
      and train_meta_difference.
    """
    metrics = dict()
    eval_until_step = basic_tools.Until(num_eval_frames)
    eval_total_steps = 0

    while eval_until_step(eval_total_steps):
        if yield_before_reset:
            yield environment, None, eval_agent, None, metrics

        t = 0
        eval_agent.reset()
        timestep_t = environment.reset()  # timestep_0.

        while True:  # For each step in the current episode.
            step_metrics = eval_agent.step(timestep=timestep_t, meta=meta, mask=mask, recall_gain=recall_gain)
            a_t = step_metrics["action"]
            slip_prob_threshold = slippery_modulator.sample(global_step_train)
            if np.random.uniform() < slip_prob_threshold:
                while True:
                    new_rand_action = np.random.randint(0, train_agent_action_shape)
                    if new_rand_action != a_t:
                        a_t = new_rand_action
                        break
            step_metrics["slip_prob_threshold"] = slip_prob_threshold
            metrics.update(step_metrics)
            del metrics["action"]  # remove action from metrics as it is not needed
            yield environment, timestep_t, eval_agent, a_t, metrics

            # Update t after one environment step and agent step and relabel.
            t += 1
            eval_total_steps += 1
            a_tm1 = a_t
            timestep_t = environment.step(a_tm1)

            if env_type == "miniworld":
                timestep_t = timestep_t._replace(observation=timestep_t.observation["observation"])

            if 0 < max_steps_per_episode <= t:
                assert t == max_steps_per_episode
                timestep_t = timestep_t._replace(step_type=dm_env.StepType.LAST)

            if timestep_t.last():
                unused_action_and_metrics = eval_agent.step(
                    timestep=timestep_t, meta=meta, mask=mask, recall_gain=recall_gain
                )
                metrics.update(unused_action_and_metrics)
                del metrics["action"]  # remove action from metrics as it is not needed
                yield environment, timestep_t, eval_agent, None, metrics
                break


def run_loop_slippery_non_periodic(
    eval_agent,
    environment: dm_env.Environment,
    train_agent_action_shape: int,
    max_steps_per_episode: int,
    num_eval_frames: int,
    global_step_train: int,
    slippery_modulator: modulators.NoisyAPeriodicSineModulator | modulators.OUDrift,
    yield_before_reset: bool = False,
    meta: Optional[jnp.ndarray] = None,
    env_type: Optional[str] = None,
    mask: Optional[jnp.ndarray] = None,
    recall_gain: Optional[jnp.ndarray] = None,
) -> Iterable[
    Tuple[
        dm_env.Environment,
        Optional[dm_env.TimeStep],
        Any,
        Optional[Action],
        Mapping[Text, Any],
    ]
]:
    """The standard run loop function.
    At time `t`, `t + 1` environment timesteps and `t + 1` agent steps have been
    seen in the current episode. `t` resets to `0` for the next episode.
    Args:
        eval_agent: Agent to be run, has methods `step(timestep)` and `reset()`.
        environment: Environment to run, has methods `step(action)` and `reset()`.
        episode, the episode is truncated.
        max_steps_per_episode: If positive, when time t reaches this value within an episode, the episode is truncated.
        num_eval_frames: Number of frames for evaluation
        yield_before_reset: Whether to additionally yield `(environment, None,
        agent, None)` before the agent and environment is reset at the start of
        each episode.
        meta: Optional task for the agent if it is APS variant

    Yields:
      Tuple `(environment, timestep_t, agent, a_t, metrics)` where
      `a_t = agent.step(timestep_t)`.
      and metrics contain information on state_frequency, state_action_frequency
      and train_meta_difference.
    """
    metrics = dict()
    eval_until_step = basic_tools.Until(num_eval_frames)
    eval_total_steps = 0

    while eval_until_step(eval_total_steps):
        if yield_before_reset:
            yield environment, None, eval_agent, None, metrics

        t = 0
        eval_agent.reset()
        timestep_t = environment.reset()  # timestep_0.

        while True:  # For each step in the current episode.
            step_metrics = eval_agent.step(timestep=timestep_t, meta=meta, mask=mask, recall_gain=recall_gain)
            a_t = step_metrics["action"]
            slip_prob_threshold = slippery_modulator.sample()
            if np.random.uniform() < slip_prob_threshold:
                while True:
                    new_rand_action = np.random.randint(0, train_agent_action_shape)
                    if new_rand_action != a_t:
                        a_t = new_rand_action
                        break
            step_metrics["slip_prob_threshold"] = slip_prob_threshold
            metrics.update(step_metrics)
            del metrics["action"]  # remove action from metrics as it is not needed
            yield environment, timestep_t, eval_agent, a_t, metrics

            # Update t after one environment step and agent step and relabel.
            t += 1
            eval_total_steps += 1
            a_tm1 = a_t
            timestep_t = environment.step(a_tm1)

            if env_type == "miniworld":
                timestep_t = timestep_t._replace(observation=timestep_t.observation["observation"])

            if 0 < max_steps_per_episode <= t:
                assert t == max_steps_per_episode
                timestep_t = timestep_t._replace(step_type=dm_env.StepType.LAST)

            if timestep_t.last():
                unused_action_and_metrics = eval_agent.step(
                    timestep=timestep_t, meta=meta, mask=mask, recall_gain=recall_gain
                )
                metrics.update(unused_action_and_metrics)
                del metrics["action"]  # remove action from metrics as it is not needed
                yield environment, timestep_t, eval_agent, None, metrics
                break
