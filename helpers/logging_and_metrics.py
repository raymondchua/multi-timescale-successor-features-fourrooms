from __future__ import annotations
from typing import Any, Iterable, Mapping, Optional, Text, Tuple, Union, Callable
import chex
import jax.numpy as jnp
import os

Action = int
Array = chex.Array
BaseAgent = Any


def prep_metrics(
    eval_with_fixed_start_pos: bool,
    fps: float,
    global_step: int,
    mode: str,
    start_pos_idx: int,
    stats: Mapping[str, Any],
    task_id: int,
    total_episodes: int,
    total_returns: float,
    total_returns_current_task: float,
    env_type: str = Union["atari", "minigrid"],
    prefix: str = "",
) -> Mapping[str, Any]:
    metrics = dict()
    if mode == "eval" and env_type == "minigrid":
        if eval_with_fixed_start_pos:
            assert start_pos_idx is not None
            metrics[prefix + "start_pos_idx"] = start_pos_idx
            metrics[prefix + "steps_to_good_policy"] = stats["steps_to_good_policy"]

    metrics[prefix + "avg_episode_returns"] = stats["mean_episode_return"]
    metrics[prefix + "episodes_done"] = stats["episodes_done"]
    metrics[prefix + "avg_episode_length"] = (
        stats["num_steps_over_episodes"] / stats["episodes_done"]
    )
    # metrics[prefix + "exploration_epsilon"] = stats["exploration_epsilon"]
    metrics[prefix + "total_returns"] = total_returns
    metrics[prefix + "total_returns_current_task"] = total_returns_current_task
    metrics[prefix + "total_episodes"] = total_episodes
    metrics[prefix + "task"] = task_id
    metrics[prefix + "steps"] = global_step
    metrics[prefix + "fps"] = fps

    # if keys in stats are not in metrics, then add them to metrics
    for key in stats.keys():
        if key not in metrics:
            # ignore if the key is timer
            if key != "timer":
                prefix_key = prefix + key
                metrics[prefix_key] = stats[key]

    return metrics


def log_stats(
    logger,
    stats: Mapping[str, Any],
    mode: str,
    task_id: int,
    global_step: int,
    total_returns_current_task: float,
    total_returns: float,
    total_episodes: int,
    fps: float,
    env_type: str,
    eval_with_fixed_start_pos: bool = False,
    start_pos_idx: Optional[int] = None,
) -> None:
    """
    Log stats using logger. During training, the prefix is an empty string. The metric is log to wandb and screen using
    the dump function. During eval, the prefix is str(task_id) + "_" + str(start_pos_idx). This prefix is used to
    generate the metric_wandb which is used to log to wandb. The original metric is only logged to screen using the
    dump_to_console function.
    Args
    ----------
    logger: Logger that has a `log_metrics` and `dump` method
    stats: Dictionary of stats to log
    mode: train or eval
    task_id: Task id can be either the current training task or the current eval task
    global_step:Global step
    total_returns_current_task: Total returns for the current task
    total_returns: Total return
    total_episodes: Total episodes
    fps: Frames per second
    env_type: Environment type
    start_pos_idx: Starting position index, only used for minigrid environment
    eval_with_fixed_start_pos: If true, then the evaluation is done with a fixed start position. Only used for minigrid

    Returns
    -------
    None
    """

    assert mode == "train" or mode == "eval"
    prefix = ""

    # check that stats do not contain nan values or non-scalar values
    for key, value in stats.items():
        assert jnp.isscalar(value), f"Value for key {key} is not scalar: {value}, type: {type(value)}"

    if mode == "eval":
        if env_type == "minigrid" and eval_with_fixed_start_pos:
            prefix = "Task: {task}, Start Pos: {start_pos}, ".format(
                task=task_id, start_pos=start_pos_idx
            )

    metrics = prep_metrics(
        env_type=env_type,
        eval_with_fixed_start_pos=eval_with_fixed_start_pos,
        fps=fps,
        global_step=global_step,
        mode=mode,
        prefix=prefix,
        start_pos_idx=start_pos_idx,
        stats=stats,
        task_id=task_id,
        total_episodes=total_episodes,
        total_returns=total_returns,
        total_returns_current_task=total_returns_current_task,
    )

    # dump to wandb
    logger.log_metrics(metrics, global_step, ty=mode)
    logger.dump_to_wandb(global_step, ty=mode)
    logger.dump_to_console(global_step, ty=mode)
    logger.clear(ty=mode)


# save jax numpy matrix into a numpy file
def save_matrix(filename: str, filepath: str, matrix: Array):
    fullpath = os.path.join(filepath, filename)

    # if the directory does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    jnp.save(fullpath, matrix)
