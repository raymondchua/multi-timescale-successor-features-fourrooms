from typing import TypeVar, TypedDict, Union
from abc import ABC, abstractmethod


class BaseDomainKwargs(TypedDict):
    action_repeat: int
    discount: float
    domain_id: int
    env_type: str
    environment_height: int
    environment_width: int
    exploration_epsilon_decay_steps: int
    impala_conv_scale: int
    max_returns: Union[float, list[float]]
    min_returns: Union[float, list[float]]
    num_stacked_frames: int
    num_unique_envs: int
    num_unique_tasks: int
    project_name: str
    seed: int
    use_framestack: bool
    use_impala_encoder: bool


DomainKwargsType = TypeVar("DomainKwargsType", bound=BaseDomainKwargs)


class BaseDomain(ABC):
    def __init__(self, **kwargs: DomainKwargsType):
        pass

    @abstractmethod
    def get_env(self, task_id: int):
        ...

    @abstractmethod
    def reset(self, **kwargs):
        ...

    @property
    @abstractmethod
    def action_repeat(self):
        ...

    @abstractmethod
    def action_spec(self):
        ...

    @abstractmethod
    def observation_spec(self):
        ...

    @abstractmethod
    def preprocessor(self, **kwargs):
        ...
    @property
    @abstractmethod
    def min_returns(self) -> Union[float, int]:
        ...

    @property
    @abstractmethod
    def max_returns(self) -> Union[float, int]:
        ...
