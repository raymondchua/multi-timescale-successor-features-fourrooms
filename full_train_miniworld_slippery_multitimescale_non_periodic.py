from workspace import MiniworldWorkspace
import hydra
import jax.numpy as jnp
from helpers import (
    training_loop_slippery_fix,
    eval_loop_slippery_fix,
    trackers,
    logging_and_metrics,
    modulators,
)
from tqdm import tqdm
import itertools
import numpy as np

from absl import logging
import wandb
import os


def train(workspace):
    logging.info("Starting training with non-periodic modulator")
    train_agent = workspace.train_agent
    eval_agent = workspace.eval_agent
    cfg = workspace.config

    train_task = workspace.train_task
    logger = workspace.logger
    timer = workspace.timer

    num_train_frames = cfg.domain.train.num_train_frames
    num_epochs = num_train_frames // cfg.domain.eval.eval_every_frames

    assert num_epochs != 0

    num_train_frames_per_epoch = num_train_frames // num_epochs
    num_train_episode_done = 0
    num_eval_episode_done = 0
    meta_update_step = 0
    steps_done_in_current_task = 0
    total_returns = 0

    if train_agent.consolidation:
        train_agent.num_train_frames = num_train_frames
        train_agent.set_up_consolidation_system()

    num_tasks_in_total = cfg.domain.num_unique_tasks * cfg.train.num_exposure

    if cfg.train.save_coverage_frequency_factor > 0:
        save_coverage_frequency = (
            num_train_frames_per_epoch // cfg.train.save_coverage_frequency_factor
        )
    else:
        save_coverage_frequency = None

    logging.info(
        "Num of tasks in total: {num_tasks_in_total}".format(
            num_tasks_in_total=num_tasks_in_total
        )
    )

    # load the preprocessor into the train agent and eval agent.
    # since it is the same preprocessor for all minigrid environments, we can load it here.
    train_agent.preprocessors(train_task.preprocessor())
    eval_agent.preprocessors(train_task.preprocessor())

    plasticity_injection_step = int(
        num_train_frames
        * cfg.train.plasticity_injection
        * cfg.domain.num_unique_tasks
        * cfg.train.num_exposure
    )

    logging.info("Plasticity injection step: {}".format(plasticity_injection_step))

    slippery_modulator = modulators.NoisyAPeriodicSineModulator(
        period=500_000,
        phase=0,
        seed=1,
        min_val=0.0,
        max_val=cfg.domain.slippery_prob,
        half_period_jitter=0.8,
        noise_std=0.05,
    )

    # slippery_vals = []
    #
    # for _ in range(num_train_frames):
    #     slippery_vals.append(slippery_modulator.sample())

    if workspace.agent_type == "sf":
        meta = train_agent.init_meta()
    else:
        meta = None

    for task_idx in tqdm(range(num_tasks_in_total)):

        unique_task_id = task_idx % cfg.domain.num_unique_tasks
        total_returns_train_current_task = (
            0  # reset train total returns for the current task
        )
        total_returns_eval_current_task = (
            0  # reset eval total returns for the current task
        )
        evaluate_task_during_training = cfg.eval.evaluate_task_during_training

        # explore only when agent first encounter the task
        if cfg.train.explore_task_on_only_first_encounter:
            if task_idx < cfg.domain.num_unique_tasks:
                train_agent.reset_exploration_frame_counter()

        # restart the epsilon-greedy policy for every new task
        elif cfg.train.explore_every_task:
            train_agent.reset_exploration_frame_counter()

        if (
            not cfg.train.explore_task_on_only_first_encounter
            and not cfg.train.explore_every_task
        ):
            logging.info(
                "Agent only explore for the first task in the first exposure and not for the remaining tasks"
            )

        if cfg.replay.reset_buffer_before_each_task:
            train_agent.reset_replay_buffer()
            steps_done_in_current_task = 0

        if cfg.train.reset_optimizer_before_each_task:
            train_agent.reset_optimizer()

        if cfg.train.use_predefined_task_id:
            assert cfg.train.predefined_task_id is not None
            logging.info(
                "Using predefined task id: {task_id}".format(
                    task_id=cfg.train.predefined_task_id
                )
            )
            current_train_id = cfg.train.predefined_task_id
        else:
            logging.info("Using task id: {task_id}".format(task_id=task_idx))
            current_train_id = task_idx

        for _ in range(num_epochs):
            train_seq = training_loop_slippery_fix.run_loop_slippery_non_periodic(
                agent=train_agent,
                environment=train_task.get_env(task_id=current_train_id),
                discount=cfg.domain.discount,
                meta_update_step=meta_update_step,
                global_step_train=workspace.global_step_train,
                learn_period=cfg.train.learn_period,
                max_steps_per_episode=workspace.max_steps_per_episode,
                meta=meta,
                num_seed_frames=cfg.train.num_seed_frames,
                num_train_frames=num_train_frames_per_epoch,
                steps_done_in_current_task=steps_done_in_current_task,
                task_id=current_train_id,
                env_type=cfg.domain.env_type,
                save_coverage_frequency=save_coverage_frequency,
                output_dir=cfg.snapshot.snapshot_dir,
                slippery_modulator=slippery_modulator,
                train_agent_action_shape=cfg.agent.action_shape,
            )

            # only slice when the num train frames per task is reached
            train_seq_truncated = itertools.islice(
                train_seq, num_train_frames_per_epoch
            )
            train_trackers = trackers.make_default_trackers(
                cfg=cfg,
                initial_agent=train_agent,
                timer=timer,
                mode="train",
            )
            train_stats = trackers.generate_statistics(
                train_trackers,
                train_seq_truncated,
            )

            workspace.global_step_train = num_train_frames_per_epoch
            steps_done_in_current_task += num_train_frames_per_epoch
            num_train_episode_done += train_stats["episodes_done"]
            timer = train_stats["timer"]
            meta_update_step = train_stats["meta_update_step"]

            train_stats["steps"] = workspace.global_step_train
            train_stats["total_time"] = timer.total_time()
            total_returns_train_current_task += train_stats["total_returns"]
            total_returns += train_stats["total_returns"]

            # accumulate stats for computing area under the curve
            workspace.avg_returns_across_tasks.append(
                train_stats["mean_episode_return"]
            )
            workspace.total_returns_across_tasks.append(train_stats["total_returns"])
            workspace.steps_done_across_tasks.append(workspace.global_step_train)

            train_stats["slip_prob"] = slippery_modulator.sample(
            )

            # remove items that are not needed for logging
            del train_stats["timer"]

            logging_and_metrics.log_stats(
                logger,
                stats=train_stats,
                mode="train",
                task_id=current_train_id,
                global_step=workspace.global_step_train,
                total_returns_current_task=total_returns_train_current_task,
                total_returns=total_returns,
                total_episodes=num_train_episode_done,
                fps=train_stats["fps"],
                env_type=cfg.domain.env_type,
            )

            if workspace.agent_type == "sf":
                meta = train_agent.meta
            else:
                meta = None

            if train_agent.has_attention_mechanism:
                eval_agent.network_params = train_agent.network_params
                eval_agent.attention_network_params = (
                    train_agent.attention_network_params
                )
                mask = train_agent.mask
                recall_gain = train_agent.recall_gain

            else:
                eval_agent.online_params = train_agent.online_params
                mask = None
                recall_gain = None

            eval_agent.reset()

            eval_seq = eval_loop_slippery_fix.run_loop_slippery_non_periodic(
                eval_agent=eval_agent,
                environment=train_task.get_env(task_id=current_train_id),
                max_steps_per_episode=workspace.max_steps_per_episode,
                num_eval_frames=cfg.domain.eval.num_eval_frames,
                meta=meta,
                mask=mask,
                recall_gain=recall_gain,
                global_step_train=workspace.global_step_train,
                train_agent_action_shape=cfg.agent.action_shape,
                slippery_modulator=slippery_modulator,
            )

            eval_seq_truncated = itertools.islice(
                eval_seq, cfg.domain.eval.num_eval_frames
            )
            eval_trackers = trackers.make_default_trackers(
                cfg=cfg, initial_agent=eval_agent, timer=timer, mode="eval"
            )
            eval_stats = trackers.generate_statistics(eval_trackers, eval_seq_truncated)

            total_returns_eval_current_task += eval_stats["total_returns"]
            num_eval_episode_done += eval_stats["episodes_done"]
            eval_stats["total_time"] = timer.total_time()

            del eval_stats["timer"]

            logging_and_metrics.log_stats(
                logger,
                stats=eval_stats,
                mode="eval",
                task_id=current_train_id,
                global_step=workspace.global_step_train,
                total_returns_current_task=total_returns_eval_current_task,
                total_returns=total_returns,
                total_episodes=num_eval_episode_done,
                fps=eval_stats["fps"],
                env_type=cfg.domain.env_type,
            )

            if (
                train_agent.use_plasticity_injection
                and workspace.global_step_train == plasticity_injection_step
            ):
                logging.log_first_n(logging.INFO, "Using plasticity injection", n=1)
                train_agent.pi_enabled()

            if cfg.snapshot.save_snapshot_when_eval:
                # add global_step_train to workspace.snapshot_dir
                snapshot_dir_train_steps = os.path.join(
                    workspace.snapshot_dir,
                    "train_step_" + str(workspace.global_step_train),
                )

                train_agent.save_snapshot(
                    task_id=current_train_id,
                    snapshot_dir=snapshot_dir_train_steps,
                    step=workspace.global_step_train,
                    cfg=cfg,
                )

        if cfg.snapshot.save_snapshot_after_each_task:
            train_agent.save_snapshot(
                task_id=current_train_id,
                snapshot_dir=workspace.snapshot_dir,
                step=workspace.global_step_train,
                cfg=cfg,
            )

        if cfg.train.terminate_after_first_task:
            print("meta: ", meta)
            break

    print("ok so far!")


@hydra.main(
    config_path=".", config_name="full_train_miniworld_slippery_fix", version_base=None
)
def main(cfg):

    if cfg.domain.env_type == "miniworld":
        workspace = MiniworldWorkspace(cfg)
    else:
        raise ValueError("Unknown environment type")
    train(workspace)


if __name__ == "__main__":
    main()
