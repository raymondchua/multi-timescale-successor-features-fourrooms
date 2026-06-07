import timeit
from typing import Any, Iterable, Mapping, Optional, Text, Tuple, Union, Callable

import collections
import dm_env
import jax
import numpy as np

from helpers import logger

Action = int
Logger = logger.Logger

BaseAgent = Any


class EpisodeTracker:
    """Tracks episode return and other statistics."""

    def __init__(
        self,
        cfg,
        timer,
    ):
        """Initializes the tracker.
        Args:
            cfg: Configuration dictionary.
            timer: Timer to use for timing.
        """
        self._num_steps_since_reset = 0
        self._num_steps_over_episodes = None
        self._episode_returns = None
        self._current_episode_rewards = None
        self._current_episode_step = None
        self._cfg = cfg
        self._timer = timer
        self._episode_done_in_current_epoch = 0
        self._finetune_train_meta_update_step = 0
        self._meta_differences = []
        self._avg_meta_difference = 0
        self._exploration_eps = np.nan
        self._elapsed_time, self._total_time = self._timer.reset()
        self._fps = 0.0
        self._output = dict()
        self._running_metrics = dict()
        self._task_grads_norm = None
        self._task_update_norm = None

    def step(
        self,
        environment: Optional[dm_env.Environment],
        timestep_t: dm_env.TimeStep,
        agent: Optional[BaseAgent],
        a_t: Optional[Action],
        metrics_t: dict[str, Any],
    ) -> None:
        """Accumulates statistics from timestep.
        Args:
            environment: Environment to step in.
            timestep_t: Timestep to accumulate statistics from.
            agent: Agent to use for stepping.
            a_t: Action to take.
            metrics_t: Metrics to log.

        Return:
            None
        """
        del (environment, agent, a_t)

        # if key is in output, overwrite the value in output
        # else add the key and value to output
        for key, value in metrics_t.items():
            if key in self._running_metrics:
                self._running_metrics[key] = value
            else:
                self._running_metrics.update({key: value})

            if key == "task_grads_norm_w":
                self._task_grads_norm.append(value)

            if key == "task_updates_norm_w":
                self._task_update_norm.append(value)

        if timestep_t.first():
            if self._current_episode_rewards:
                raise ValueError("Current episode reward list should be empty.")
            if self._current_episode_step != 0:
                raise ValueError("Current episode step should be zero.")
        else:
            # First reward is invalid, all other rewards are appended.
            self._current_episode_rewards.append(timestep_t.reward)

        if "train_meta_difference" in metrics_t:
            self._meta_differences.append(metrics_t["train_meta_difference"])

        self._num_steps_since_reset += 1
        self._current_episode_step += 1
        episode_frame = self._current_episode_step * self._cfg.domain.action_repeat

        if "finetune_train_meta_update_step" in metrics_t:
            self._finetune_train_meta_update_step = metrics_t[
                "finetune_train_meta_update_step"
            ]
        if "exploration_epsilon" in metrics_t:
            self._exploration_eps = metrics_t["exploration_epsilon"]

        if timestep_t.last():
            self._episode_done_in_current_epoch += 1
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._num_steps_over_episodes += episode_frame

            self._current_episode_rewards = []
            self._current_episode_step = 0

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._num_steps_over_episodes = 0
        self._episode_returns = []
        self._current_episode_step = 0
        self._current_episode_rewards = []
        self._episode_done_in_current_epoch = 0
        self._meta_differences = []
        self._avg_meta_difference = 0
        self._fps = 0.0
        self._running_metrics = dict()
        self._task_grads_norm = []
        self._task_update_norm = []

    def get(self) -> dict[str, Union[int, float, None]]:
        """Aggregates statistics and returns as a dictionary.

        Here the convention is `episode_return` is set to `current_episode_return`
        if a full episode has not been encountered. Otherwise it is set to
        `mean_episode_return` which is the mean return of complete episodes only. If
        no steps have been taken at all, `episode_return` is set to `NaN`.

        Returns:
          A dictionary of aggregated statistics.
        """
        if self._episode_returns:
            mean_episode_return = np.array(self._episode_returns).mean()
            current_episode_return = sum(self._current_episode_rewards)
            episode_return = mean_episode_return
            total_returns = sum(self._episode_returns)
        else:
            mean_episode_return = np.nan
            if self._num_steps_since_reset > 0:
                current_episode_return = sum(self._current_episode_rewards)
            else:
                current_episode_return = np.nan
            episode_return = current_episode_return
            total_returns = sum(self._episode_returns)

        if self._task_grads_norm:
            mean_task_grads_norm = np.array(self._task_grads_norm).mean()

        else:
            mean_task_grads_norm = np.nan

        if self._task_update_norm:
            mean_task_update_norm = np.array(self._task_update_norm).mean()

        else:
            mean_task_update_norm = np.nan

        self._elapsed_time, self._total_time = self._timer.reset()
        self._fps = self._num_steps_since_reset / self._elapsed_time
        self._avg_meta_difference = np.mean(self._meta_differences)

        temp = {
            "mean_episode_return": mean_episode_return,
            "current_episode_return": current_episode_return,
            "episode_return": episode_return,
            "episodes_done": self._episode_done_in_current_epoch,
            "total_returns": total_returns,
            "num_steps_over_episodes": self._num_steps_over_episodes,
            "current_episode_step": self._current_episode_step,
            "num_steps_since_reset": self._num_steps_since_reset,
            "timer": self._timer,
            "finetune_train_meta_update_step": self._finetune_train_meta_update_step,
            "fps": self._fps,
            "avg_meta_differences": self._avg_meta_difference,
            "exploration_epsilon": self._exploration_eps,
            "mean_task_grads_norm": mean_task_grads_norm,
            "mean_task_update_norm": mean_task_update_norm,
        }

        # add running metrics to output if key is not in output. Ignore if key is meta
        for key, value in self._running_metrics.items():
            if key not in temp and key != "meta":
                temp[key] = value

        return temp


class EvalEpisodeTracker:
    """Tracks episode return and other statistics."""

    def __init__(
        self,
        cfg,
        timer,
    ):
        """Initializes the tracker.

        Args:
            cfg: The config.
            timer: The timer.

        """
        self._num_steps_since_reset = None
        self._num_steps_over_episodes = None
        self._episode_returns = None
        self._episode_pred_returns = None
        self._current_episode_rewards = None
        self._current_episode_pred_rewards = None
        self._current_episode_step = None
        self._cfg = cfg
        self._timer = timer
        self._eval_episode_done_in_current_epoch = 0
        self._fps = 0.0
        self._exploration_eps = np.nan
        self._elapsed_time, self._total_time = self._timer.reset()
        self._output = dict()
        self._running_metrics = dict()

    def step(
        self,
        environment: Optional[dm_env.Environment],
        timestep_t: dm_env.TimeStep,
        agent: Optional[BaseAgent],
        a_t: Optional[Action],
        metrics_t: Mapping[Text, Any],
    ) -> None:
        """Accumulates statistics from timestep.

        Args:
            environment: The environment.
            timestep_t: The timestep.
            agent: The agent.
            a_t: The action.
            metrics_t: The metrics from the agent.
        """
        del (environment, agent, a_t)

        # if key is in output, overwrite the value in output
        # else add the key and value to output
        for key, value in metrics_t.items():
            if key in self._running_metrics:
                self._running_metrics[key] = value
            else:
                self._running_metrics.update({key: value})

        if timestep_t.first():
            if self._current_episode_rewards:
                raise ValueError("Current episode reward list should be empty.")
            if self._current_episode_step != 0:
                raise ValueError("Current episode step should be zero.")
        else:
            # First reward is invalid, all other rewards are appended.
            self._current_episode_rewards.append(timestep_t.reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if "exploration_epsilon" in metrics_t:
            self._exploration_eps = metrics_t["exploration_epsilon"]

        if timestep_t.last():
            self._eval_episode_done_in_current_epoch += 1
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._episode_pred_returns.append(sum(self._current_episode_pred_rewards))
            self._current_episode_rewards = []
            self._current_episode_pred_rewards = []
            self._num_steps_over_episodes += self._current_episode_step
            self._current_episode_step = 0

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._num_steps_over_episodes = 0
        self._episode_returns = []
        self._episode_pred_returns = []
        self._current_episode_step = 0
        self._current_episode_rewards = []
        self._current_episode_pred_rewards = []
        self._eval_episode_done_in_current_epoch = 0
        self._running_metrics = dict()
        self._fps = 0.0

    def get(self) -> Mapping[Text, Union[int, float, None]]:
        """Aggregates statistics and returns as a dictionary.

        Here the convention is `episode_return` is set to `current_episode_return`
        if a full episode has not been encountered. Otherwise it is set to
        `mean_episode_return` which is the mean return of complete episodes only. If
        no steps have been taken at all, `episode_return` is set to `NaN`.

        Returns:
          A dictionary of aggregated statistics.
        """
        if self._episode_returns:
            mean_episode_return = np.array(self._episode_returns).mean()
            mean_episode_pred_return = np.array(self._episode_pred_returns).mean()
            current_episode_return = sum(self._current_episode_rewards)
            current_episode_pred_return = sum(self._current_episode_pred_rewards)
            episode_return = mean_episode_return
            episode_pred_return = mean_episode_pred_return
            total_returns = sum(self._episode_returns)
        else:
            mean_episode_return = np.nan
            if self._num_steps_since_reset > 0:
                current_episode_return = sum(self._current_episode_rewards)
                current_episode_pred_return = sum(self._current_episode_pred_rewards)
            else:
                current_episode_return = np.nan
                current_episode_pred_return = np.nan
            episode_return = current_episode_return
            episode_pred_return = current_episode_pred_return
            total_returns = sum(self._episode_returns)

        self._elapsed_time, self._total_time = self._timer.reset()
        self._fps = self._num_steps_since_reset / self._elapsed_time

        temp = {
            "mean_episode_return": mean_episode_return,
            "current_episode_return": current_episode_return,
            "current_episode_pred_return": current_episode_pred_return,
            "episode_return": episode_return,
            "episode_pred_return": episode_pred_return,
            "episodes_done": self._eval_episode_done_in_current_epoch,
            "total_returns": total_returns,
            "num_steps_over_episodes": self._num_steps_over_episodes,
            "current_episode_step": self._current_episode_step,
            "num_steps_since_reset": self._num_steps_since_reset,
            "fps": self._fps,
            "timer": self._timer,
        }
        # add running metrics to output if key is not in output. Ignore if key is meta
        for key, value in self._running_metrics.items():
            if key not in temp and key != "meta":
                temp[key] = value

        return temp


class SimpleTracker:
    """A simple and clean tracker. Tracks episode return and number of episodes."""

    def __init__(
        self,
        cfg,
        good_policy_threshold_for_current_task: int,
        train_steps_completed: int,
    ):
        """Initializes the tracker.

        Args:
            cfg: The config.
            good_policy_threshold_for_current_task: The good policy threshold for the current task.
            train_steps_completed: The number of training steps completed for current task.
        """
        self._num_steps_since_reset = None
        self._num_steps_over_episodes = None
        self._episode_returns = None
        self._current_episode_rewards = None
        self._current_episode_step = None
        self._cfg = cfg
        self._episodes_done = 0  # number of episodes done in current epoch
        self._exploration_eps = np.nan
        self._good_policy_threshold_for_current_task = (
            good_policy_threshold_for_current_task
        )
        self._moving_avg_episode_length = np.zeros((self._cfg["moving_avg_episodes"],))
        self._steps_to_good_policy = np.inf  # means no good policy found yet
        self._good_policy_threshold = self._cfg["good_policy_threshold"]
        self._num_moving_avg_episodes = self._cfg["moving_avg_episodes"]
        self._train_steps_completed = train_steps_completed

    def step(
        self,
        environment: Optional[dm_env.Environment],
        timestep_t: dm_env.TimeStep,
        agent: Optional[BaseAgent],
        a_t: Optional[Action],
        metrics_t: Mapping[Text, Any],
    ) -> None:
        """Accumulates statistics from timestep.

        Args:
            environment: The environment.
            timestep_t: The timestep.
            agent: The agent.
            a_t: The action.
            metrics_t: The metrics from the agent.
        """
        del (environment, agent, a_t)

        if timestep_t.first():
            if self._current_episode_rewards:
                raise ValueError("Current episode reward list should be empty.")
            if self._current_episode_step != 0:
                raise ValueError("Current episode step should be zero.")
        else:
            # First reward is invalid, all other rewards are appended.
            self._current_episode_rewards.append(timestep_t.reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if "exploration_epsilon" in metrics_t:
            self._exploration_eps = metrics_t["exploration_epsilon"]

        if timestep_t.last():
            self._episodes_done += 1
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._current_episode_rewards = []
            self._num_steps_over_episodes += self._current_episode_step
            self._current_episode_step = 0

            # if the overall episode count is at least the number of episodes to average over
            # compute the moving average of the episode length
            # if the moving average is less than or equals to the good policy threshold,
            if self._episodes_done >= self._num_moving_avg_episodes:
                avg_steps_current_policy = np.mean(self._moving_avg_episode_length)
                if avg_steps_current_policy <= self._good_policy_threshold:
                    self._steps_to_good_policy = self._train_steps_completed

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._num_steps_over_episodes = 0
        self._episode_returns = []
        self._current_episode_step = 0
        self._current_episode_rewards = []
        self._episodes_done = 0
        self._steps_to_good_policy = np.inf
        self._exploration_eps = np.nan

    def get(self) -> Mapping[Text, Union[int, float, None]]:
        """Aggregates statistics and returns as a dictionary.

        Here the convention is `episode_return` is set to `current_episode_return`
        if a full episode has not been encountered. Otherwise, it is set to
        `mean_episode_return` which is the mean return of complete episodes only. If
        no steps have been taken at all, `episode_return` is set to `NaN`.

        Returns:
          A dictionary of aggregated statistics.
        """
        if self._episode_returns:
            mean_episode_return = np.array(self._episode_returns).mean()
            episode_return = mean_episode_return
            total_returns = sum(self._episode_returns)
        else:
            mean_episode_return = np.nan
            if self._num_steps_since_reset > 0:
                current_episode_return = sum(self._current_episode_rewards)
            else:
                current_episode_return = np.nan
            episode_return = current_episode_return
            total_returns = sum(self._episode_returns)

        return {
            "mean_episode_return": mean_episode_return,
            "episode_return": episode_return,
            "num_episodes": self._episodes_done,
            "total_returns": total_returns,
            "num_steps_over_episodes": self._num_steps_over_episodes,
            "current_episode_step": self._current_episode_step,
            "num_steps_since_reset": self._num_steps_since_reset,
            "steps_to_good_policy": self._steps_to_good_policy,
            "exploration_epsilon": self._exploration_eps,
        }


class StepRateTracker:
    """Tracks step rate, number of steps taken and duration since last reset."""

    def __init__(self):
        self._num_steps_since_reset = None
        self._start = None

    def step(
        self,
        environment: Optional[dm_env.Environment],
        timestep_t: Optional[dm_env.TimeStep],
        agent: Optional[BaseAgent],
        a_t: Optional[Action],
        metrics_t: Mapping[Text, Any],
    ) -> None:
        del (
            environment,
            timestep_t,
            agent,
            a_t,
            metrics_t,
        )
        self._num_steps_since_reset += 1

    def reset(self) -> None:
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            "step_rate": step_rate,
            "num_steps": self._num_steps_since_reset,
            "duration": duration,
        }


class EvalStepRateTracker:
    """Tracks step rate, number of steps taken and duration since last reset."""

    def __init__(self):
        self._num_steps_since_reset = None
        self._start = None

    def step(
        self,
        environment: Optional[dm_env.Environment],
        timestep_t: Optional[dm_env.TimeStep],
        agent: Optional[BaseAgent],
        a_t: Optional[Action],
        metrics_t: Mapping[Text, Any],
    ) -> None:
        del (environment, timestep_t, agent, a_t, metrics_t)
        self._num_steps_since_reset += 1

    def reset(self) -> None:
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            "step_rate": step_rate,
            "num_steps": self._num_steps_since_reset,
            "duration": duration,
        }


class UnbiasedExponentialWeightedAverageAgentTracker:
    """'Unbiased Constant-Step-Size Trick' from the Sutton and Barto RL book."""

    def __init__(self, step_size: float, initial_agent: BaseAgent):
        self._initial_statistics = dict(initial_agent.statistics)
        self._step_size = step_size
        self.trace = 0.0
        self._statistics = dict(self._initial_statistics)

    def step(
        self,
        environment: Optional[dm_env.Environment],
        timestep_t: Optional[dm_env.TimeStep],
        agent: BaseAgent,
        a_t: Optional[Action],
        metrics_t: Mapping[Text, Any],
    ) -> None:
        """Accumulates agent statistics."""
        del (environment, timestep_t, a_t, metrics_t)

        self.trace = (1 - self._step_size) * self.trace + self._step_size
        final_step_size = self._step_size / self.trace
        assert 0 <= final_step_size <= 1

        if final_step_size == 1:
            # Since the self._initial_statistics is likely to be NaN and
            # 0 * NaN == NaN just replace self._statistics on the first step.
            self._statistics = dict(agent.statistics)
        else:
            self._statistics = jax.tree_util.tree_map(
                lambda s, x: (1 - final_step_size) * s + final_step_size * x,
                self._statistics,
                agent.statistics,
            )

    def reset(self) -> None:
        """Resets statistics and internal state."""
        self.trace = 0.0
        # get() may be called before step() so ensure statistics are initialized.
        self._statistics = dict(self._initial_statistics)

    def get(self) -> Mapping[Text, float]:
        """Returns current accumulated statistics."""
        return self._statistics


def make_default_trackers(
    cfg: dict,
    initial_agent: BaseAgent,
    timer,
    mode: str,
):

    assert mode in ["train", "eval"]

    if mode == "train":
        episode_tracker = EpisodeTracker(cfg=cfg, timer=timer)
    else:
        episode_tracker = EvalEpisodeTracker(cfg=cfg, timer=timer)

    return [
        episode_tracker,
        StepRateTracker(),
        UnbiasedExponentialWeightedAverageAgentTracker(
            step_size=1e-3, initial_agent=initial_agent
        ),
    ]


# make a simple tracker, which only tracks the episode returns and the number of episodes
def make_simple_trackers(
    cfg: dict,
    good_policy_threshold_for_current_task: Optional[int],
    train_steps_completed: int,
):
    return [
        SimpleTracker(
            cfg=cfg,
            good_policy_threshold_for_current_task=good_policy_threshold_for_current_task,
            train_steps_completed=train_steps_completed,
        ),
        EvalStepRateTracker(),
    ]


def generate_statistics(
    trackers: Iterable[Any],
    timestep_action_sequence: Iterable[
        Tuple[
            dm_env.Environment,
            Optional[dm_env.TimeStep],
            BaseAgent,
            Optional[Action],
            Mapping[Text, Any],
        ]
    ],
) -> Mapping[Text, Any]:
    """Generates statistics from a sequence of timestep and actions."""
    # Only reset at the start, not between episodes.
    count = 0
    for tracker in trackers:
        tracker.reset()

    for (
        environment,
        timestep_t,
        agent,
        a_t,
        metrics_t,
    ) in timestep_action_sequence:
        count += 1
        for tracker in trackers:
            tracker.step(
                environment=environment,
                timestep_t=timestep_t,
                agent=agent,
                a_t=a_t,
                metrics_t=metrics_t,
            )

    # Merge all statistics dictionaries into one.
    statistics_dicts = (tracker.get() for tracker in trackers)
    return dict(collections.ChainMap(*statistics_dicts))