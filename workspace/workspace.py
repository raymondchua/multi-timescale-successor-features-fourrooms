import warnings
import chex
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import hydra
import wandb

from abc import ABC, abstractmethod
from absl import logging
import agent.Epsilon_greedy_agent as Epsilon_greedy_agent
from agent.Epsilon_greedy_consolidation_softmax_attention_agent import (
    EpsilonGreedyConsolidationSoftmaxAttentionActor,
)
import omegaconf
from helpers import Logger, basic_tools


Array = chex.Array
EpsilonGreedyActor = Epsilon_greedy_agent.EpsilonGreedyActor
EpsilonGreedyAttentionActor = EpsilonGreedyConsolidationSoftmaxAttentionActor

Config = omegaconf.dictconfig.DictConfig


def make_agent(cfg: Config):
    return hydra.utils.instantiate(cfg)


def make_task(cfg: Config, seed: int):
    cfg.seed = seed
    return hydra.utils.instantiate(cfg)


class BaseWorkspace(ABC):
    """
    Workspace
    """

    def __init__(self, cfg: Config, **kwargs):
        self.work_dir = Path.cwd()
        self._cfg = cfg

        logging.log_first_n(logging.INFO, "Loading workspace...", 1)

        basic_tools.set_seed_everywhere(
            self._cfg.train.seed
        )  # Set seed for all randomness sources

        # determine the num of stacked frames for the environment
        if self._cfg.domain.use_framestack:
            if self._cfg.domain.env_type == "minatar":
                # this is based on the max number of channels among all the minatar environments.
                self._cfg.domain.num_stacked_frames = 10
            else:
                self._cfg.domain.num_stacked_frames = (
                    4  # stack the last 4 frames following DQN
                )

        else:
            self._cfg.domain.num_stacked_frames = 3  # set to 3 to use RGB frame

        self._training_task = make_task(self._cfg.domain, self._cfg.train.seed)
        self._action_shape = self.train_task.action_spec().num_values
        self._cfg.agent.action_shape = self._action_shape
        self._cfg.env.obs_shape = self.train_task.observation_spec().shape
        self._cfg.env.action_repeat = self.train_task.action_repeat

        self._cfg.paths.work_dir = self.work_dir
        self._cfg.train.num_seed_frames = (
            self._cfg.replay.min_replay_capacity_fraction
            * self._cfg.domain.replay.replay_buffer_size
            * self._cfg.env.action_repeat
        )
        self._cfg.train.exploration_epsilon_decay_frame_fraction = (
            self._cfg.domain.train.exploration_epsilon_decay_steps
            / self._cfg.domain.train.num_train_frames
        )

        # Ensure that the number of seed frames is equal to the time step when we start decaying the exploration epsilon
        assert (
            self._cfg.train.num_seed_frames
            == self._cfg.replay.min_replay_capacity_fraction
            * self._cfg.domain.replay.replay_buffer_size
            * self._cfg.env.action_repeat
        )

        logging.info(
            "Exploration epsilon decay frame fraction: %f",
            self._cfg.train.exploration_epsilon_decay_frame_fraction,
        )

        if self._cfg.agent.has_task_params:

            if self._cfg.domain.env_type == "minatar":
                self._cfg.agent.lr_task = self._cfg.agent.minatar.lr_task

            elif self._cfg.domain.env_type == "miniworld":
                self._cfg.agent.lr_task = self._cfg.agent.miniworld.lr_task
            elif self._cfg.domain.env_type == "minigrid":
                self._cfg.agent.lr_task = self._cfg.agent.minigrid.lr_task
            else:
                raise ValueError(
                    "Invalid environment type: {}".format(self._cfg.domain.env_type)
                )

        elif not self._cfg.agent.has_task_params:
            self._cfg.agent.lr_task = 0.0
        else:
            # raise an error if the agent has no environment specific lr_task parameter
            raise ValueError("The agent has no environment specific lr_task parameter")

        logging.info("lr_task: %f", self._cfg.agent.lr_task)

        self._agent = make_agent(
            self._cfg.agent,
        )

        if hasattr(self._agent, "init_meta"):
            self._agent_type = "sf"

        else:
            self._agent_type = "ql"

        # Create snapshot directory if save_snapshot_after_each_task is true. Otherwise, use the snapshot_dir from the
        # config file
        self._snapshot_dir = self._cfg.snapshot.base_dir

        if (
            self._cfg.domain.discount != 0.99
            and not self._cfg.snapshot.use_alternate_discount
        ):
            raise ValueError(
                "If the discount factor is not 0.99, then use_alternate_discount must be set to True in the snapshot "
                "config to ensure that the snapshot directory is different from the one with discount factor 0.99"
            )

        elif self._cfg.domain.discount != 0.99 and self._cfg.snapshot.use_alternate_discount:
            self._snapshot_dir = os.path.join(
                self._cfg.snapshot.base_dir,
                self._cfg.domain.train.logging.experiment,
                self._cfg.domain.short_name,
                self._cfg.agent.name,
                str(self._cfg.train.seed),
                str("discount_") + str(self._cfg.domain.discount),
                str("nstep_") + str(self._cfg.replay.nstep)
            )

        else:
            # add domain.train.logging.experiment to snapshot_dir as a sub-directory
            self._snapshot_dir = os.path.join(
                self._snapshot_dir,
                self._cfg.domain.train.logging.experiment,
                self._cfg.domain.short_name,
                self._cfg.agent.name,
                str(self._cfg.train.seed),
            )

        # agent config's consolidation is true
        if self._cfg.agent.consolidation or self._cfg.agent.task_consolidation:
            # add num_beakers_ agents.num_beakers to snapshot_dir as a sub-directory
            self._snapshot_dir = os.path.join(
                self._snapshot_dir,
                "num_beakers_" + str(self._cfg.agent.num_beakers),
            )

        if self._cfg.agent.name == "aps_sparsify_agent":
            decimal_reg_sf = str(self._cfg.agent.reg_sf).split(".")[1]
            self._snapshot_dir = os.path.join(
                self._snapshot_dir, "reg_sf_" + decimal_reg_sf
            )

        # logging info of the snapshot_dir that is just created
        logging.info("snapshot_dir: %s", self._snapshot_dir)

        # flatten the cfg file
        self._cfg_flatten = basic_tools.dictionary_flatten(self._cfg)

        if self._cfg.logging.mode == "full_train":
            experiment = self._cfg.domain.train.logging.experiment

        elif self._cfg.logging.mode == "test":
            experiment = self._cfg.domain.test.logging.experiment

        elif self._cfg.logging.mode == "pretrain":
            experiment = self._cfg.domain.pretrain.logging.experiment
        elif self._cfg.logging.mode == "full_train_slippery":
            experiment = self._cfg.domain.train.logging.experiment + "_slippery"

        elif self._cfg.logging.mode == "full_train_slippery_fix":
            experiment = self._cfg.domain.train.logging.experiment + "_slippery_fix"

        elif self._cfg.logging.mode == "full_train_slippery_fix_non_periodic":
            experiment = (
                self._cfg.domain.train.logging.experiment + "_slippery_fix_non_periodic"
            )

        elif self._cfg.logging.mode == "full_train_slippery_fix_ou":
            experiment = self._cfg.domain.train.logging.experiment + "_slippery_fix_ou"

        else:
            raise ValueError("Invalid logging mode: {}".format(self._cfg.logging.mode))

        # create logger
        if self._cfg.logging.use_wandb:
            exp_name = "_".join(
                [
                    experiment,
                ]
            )

            # get current working directory and add wandb_dir
            wandb_dir_absolute = Path.cwd()

            # convert wandb_dir_absolute to string
            wandb_dir_str = wandb_dir_absolute.as_posix()

            # log wandb_dir_str
            logging.info("wandb_dir_str: %s", wandb_dir_str)

            project_name = (
                "continual_rl" + "_" + self._cfg.logging.mode + "_" + experiment
            )
            wandb.init(
                project=project_name,
                group=cfg.agent.name,
                name=exp_name,
                config=self._cfg_flatten,
                dir=wandb_dir_str,
                mode=cfg.logging.wandb_mode,
                settings=wandb.Settings(
                    start_method="thread"
                ),  # required for offline mode
            )

        else:
            wandb.init(mode="disabled")

        self._logger = Logger(self.work_dir, use_wandb=cfg.logging.use_wandb)

        # set up eval agent
        if self._agent.has_attention_mechanism:
            self._eval_agent = EpsilonGreedyAttentionActor(cfg=self._cfg)
            self._eval_agent.network_params = self._agent.network_params
            self._eval_agent.attention_network = self._agent.attention_network
            self._eval_agent.attention_network_params = (
                self._agent.attention_network_params
            )

        else:
            self._eval_agent = EpsilonGreedyActor(
                cfg=self._cfg,
            )

        self._eval_agent.online_params = self._agent.online_params
        self._eval_agent.network = self._agent.eval_network
        self._eval_agent.eval_rng_key = self._agent.eval_rng_key

        self._agent.logger = self.logger

        self._timer = basic_tools.Timer()
        self._global_step_train = 0
        self._global_step_eval = 0
        self._global_episode = 0
        self._eval_episodes_done = 0
        self._steps_over_episodes = 0

        self._avg_returns_across_tasks = []
        self._total_returns_across_tasks = []
        self._steps_done_across_tasks = []

        logging.info("{}\n".format(self._cfg_flatten))

    @property
    def global_step_train(self):
        """Returns the total number of steps taken by the agent during training."""
        return self._global_step_train

    @global_step_train.setter
    def global_step_train(self, value):
        self._global_step_train += value

    @property
    def global_step_eval(self):
        return self._global_step_eval

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def decay_steps(self):
        decay_steps = int(
            self._cfg.train.exploration_epsilon_decay_frame_fraction
            * self._cfg.domain.train.num_train_frames
        )

        return decay_steps

    @property
    @abstractmethod
    def max_steps_per_episode(self):
        ...

    def load_snapshot(self, snapshot_path):
        snapshot = Path(snapshot_path)
        self._agent.load_snapshot(snapshot)

    @property
    def train_agent(self):
        return self._agent

    @property
    def eval_agent(self):
        return self._eval_agent

    @property
    def config(self):
        return self._cfg

    @property
    def train_task(self):
        return self._training_task

    @property
    def logger(self):
        return self._logger

    @property
    def timer(self):
        return self._timer

    @property
    def action_shape(self):
        return self._action_shape

    @property
    def avg_returns_across_tasks(self):
        return self._avg_returns_across_tasks

    @property
    def total_returns_across_tasks(self):
        return self._total_returns_across_tasks

    @property
    def steps_done_across_tasks(self):
        return self._steps_done_across_tasks

    @property
    def agent_type(self):
        return self._agent_type

    @property
    def snapshot_dir(self):
        return self._snapshot_dir
