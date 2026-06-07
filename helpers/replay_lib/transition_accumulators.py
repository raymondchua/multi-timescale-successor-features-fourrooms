import dm_env
import numpy as np
import collections
from typing import Iterable

from .transition_types import (
    Transition,
    Transition_SF,
    Transition_Task,
    Transition_Task_OneHotAction,
    Transition_Task_Prev_Action,
)

from .transition_builds import (
    build_n_step_SF_transition,
    build_n_step_transition,
    build_n_step_transition_task,
    build_n_step_transition_task_one_hot_action,
    build_n_step_transition_task_prev_action,
)

from .replay_helpers import (
    Action,
    Task,
)


class TransitionAccumulator:
    """Accumulates timesteps to form transitions."""

    def __init__(self):
        self.reset()

    def step(self, timestep_t: dm_env.TimeStep, a_t: Action) -> Iterable[Transition]:
        """Accumulates timestep and resulting action, maybe yield a transition."""
        if timestep_t.first():
            self.reset()

        if self._timestep_tm1 is None:
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            return  # Empty iterable.
        else:
            transition = Transition(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
            )
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            yield transition

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be `FIRST`."""
        self._timestep_tm1 = None
        self._a_tm1 = None


class TransitionSFAccumulator:
    """
    Accumulates timesteps to form transitions.
    This transition includes:
    previous state s_tm1,
    previous action a_tm1,
    reward r_t,
    discount discount_t,
    current state s_t,
    current action a_t
    """

    def __init__(self, num_actions: int):
        self.reset()
        self._num_actions = num_actions
        self._a_tm1_vector = None

    def step(
        self, timestep_t: dm_env.TimeStep, a_t: Action, task: Task
    ) -> Iterable[Transition_SF]:
        """Accumulates timestep and resulting action, maybe yield a transition."""

        if timestep_t.first():
            self.reset()

        if self._timestep_tm1 is None:
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            self._task = task
            self._a_tm1_vector = np.zeros(self._num_actions)
            self._a_tm1_vector[self._a_tm1] = 1
            return  # Empty iterable.
        else:
            a_t_vector = np.zeros(self._num_actions)
            a_t_vector[a_t] = 1
            transition = Transition_SF(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                a_tm1_vector=self._a_tm1_vector,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
                a_t_vector=a_t_vector,
                task=self._task,
            )
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            yield transition

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be `FIRST`."""
        self._timestep_tm1 = None
        self._a_tm1 = None
        self._a_tm1_vector = None
        self._task = None


class TransitionTaskOneHotActionAccumulator:
    """
    Accumulates timesteps to form transitions.
    This transition includes:
    previous state s_tm1,
    previous action a_tm1,
    reward r_t,
    discount discount_t,
    current state s_t,
    current action a_t
    """

    def __init__(self, num_actions: int):
        self.reset()
        self._num_actions = num_actions
        self._a_tm1_vector = None

    def step(
        self, timestep_t: dm_env.TimeStep, a_t: Action
    ) -> Iterable[Transition_Task_OneHotAction]:
        """Accumulates timestep and resulting action, maybe yield a transition."""

        if timestep_t.first():
            self.reset()

        if self._timestep_tm1 is None:
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            self._a_tm1_vector = np.zeros(self._num_actions)
            self._a_tm1_vector[self._a_tm1] = 1
            return  # Empty iterable.
        else:
            a_t_vector = np.zeros(self._num_actions)
            a_t_vector[a_t] = 1
            transition = Transition_Task_OneHotAction(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                a_tm1_vector=self._a_tm1_vector,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
            )
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            yield transition

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be `FIRST`."""
        self._timestep_tm1 = None
        self._a_tm1 = None
        self._a_tm1_vector = None


class TransitionTaskAccumulator:
    """Accumulates timesteps to form transitions."""

    def __init__(self, num_actions: int):
        self.reset()
        self._num_actions = num_actions

    def step(
        self,
        timestep_t: dm_env.TimeStep,
        a_t: Action,
        task: Task,
    ) -> Iterable[Transition_Task]:
        """Accumulates timestep and resulting action, maybe yield a transition."""
        if timestep_t.first():
            self.reset()

        if self._timestep_tm1 is None:
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            self._task = task
            return  # Empty iterable.
        else:
            transition = Transition_Task(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
                task=self._task,
            )
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            yield transition

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be `FIRST`."""
        self._timestep_tm1 = None
        self._a_tm1 = None
        self._task = None

class TransitionTaskPrevActionAccumulator:
    """Accumulates timesteps to form transitions."""

    def __init__(self, num_actions: int):
        self.reset()
        self._num_actions = num_actions

    def step(
        self,
        timestep_t: dm_env.TimeStep,
        a_t: Action,
        task: Task,
    ) -> Iterable[Transition_Task_Prev_Action]:
        """Accumulates timestep and resulting action, maybe yield a transition."""
        if timestep_t.first():
            self.reset()

        if self._timestep_tm1 is None:
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            self._a_tm2 = None
            self._task = task
            return  # Empty iterable.
        else:
            transition = Transition_Task_Prev_Action(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                a_tm2=self._a_tm2,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
                task=self._task,
            )
            self._timestep_tm1 = timestep_t
            self._a_tm2 = self._a_tm1
            self._a_tm1 = a_t
            yield transition

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be `FIRST`."""
        self._timestep_tm1 = None
        self._a_tm1 = None
        self._a_tm2 = None
        self._task = None


class NStepTransitionAccumulator:
    """Accumulates timesteps to form n-step transitions.
    Let `t` be the index of a timestep within an episode and `T` be the index of
    the final timestep within an episode. Then given the step type of the timestep
    passed into `step()` the accumulator will:
    *   `FIRST`: yield nothing.
    *   `MID`: if `t < n`, yield nothing, else yield one n-step transition
        `s_{t - n} -> s_t`.
    *   `LAST`: yield all transitions that end at `s_t = s_T` from up to n steps
        away, specifically `s_{T - min(n, T)} -> s_T, ..., s_{T - 1} -> s_T`.
        These are `min(n, T)`-step, ..., `1`-step transitions.
    """

    def __init__(self, n):
        self._transitions = collections.deque(maxlen=n)  # Store 1-step transitions.
        self.reset()

    def step(self, timestep_t: dm_env.TimeStep, a_t: Action) -> Iterable[Transition]:
        """Accumulates timestep and resulting action, yields transitions."""
        if timestep_t.first():
            self.reset()

        # There are no transitions on the first timestep.
        if self._timestep_tm1 is None:
            assert self._a_tm1 is None
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            return  # Empty iterable.

        self._transitions.append(
            Transition(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
            )
        )

        self._timestep_tm1 = timestep_t
        self._a_tm1 = a_t

        if timestep_t.last():
            # Yield any remaining n, n-1, ..., 1-step transitions at episode end.
            while self._transitions:
                yield build_n_step_transition(self._transitions)
                self._transitions.popleft()
        else:
            # Wait for n transitions before yielding anything.
            if len(self._transitions) < self._transitions.maxlen:
                return  # Empty iterable.

            assert len(self._transitions) == self._transitions.maxlen

            # This is the typical case, yield a single n-step transition.
            yield build_n_step_transition(self._transitions)

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be FIRST."""
        self._transitions.clear()
        self._timestep_tm1 = None
        self._a_tm1 = None


class NStepTransitionTaskAccumulator:
    """Accumulates timesteps with task to form n-step transitions.
    Let `t` be the index of a timestep within an episode and `T` be the index of
    the final timestep within an episode. Then given the step type of the timestep
    passed into `step()` the accumulator will:
    *   `FIRST`: yield nothing.
    *   `MID`: if `t < n`, yield nothing, else yield one n-step transition
        `s_{t - n} -> s_t`.
    *   `LAST`: yield all transitions that end at `s_t = s_T` from up to n steps
        away, specifically `s_{T - min(n, T)} -> s_T, ..., s_{T - 1} -> s_T`.
        These are `min(n, T)`-step, ..., `1`-step transitions.
    """

    def __init__(self, n, num_actions: int):
        self._transitions = collections.deque(maxlen=n)  # Store 1-step transitions.
        self.reset()
        self._num_actions = num_actions

    def step(
        self,
        timestep_t: dm_env.TimeStep,
        a_t: Action,
        task: Task,
    ) -> Iterable[Transition_Task]:
        """Accumulates timestep and resulting action, yields transitions."""
        if timestep_t.first():
            self.reset()

        # There are no transitions on the first timestep.
        if self._timestep_tm1 is None:
            assert self._a_tm1 is None
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            self._task = task
            return  # Empty iterable.

        self._transitions.append(
            Transition_Task(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
                task=self._task,
            )
        )

        self._timestep_tm1 = timestep_t
        self._a_tm1 = a_t

        if timestep_t.last():
            # Yield any remaining n, n-1, ..., 1-step transitions at episode end.
            while self._transitions:
                yield build_n_step_transition_task(self._transitions)
                self._transitions.popleft()
        else:
            # Wait for n transitions before yielding anything.
            if len(self._transitions) < self._transitions.maxlen:
                return  # Empty iterable.

            assert len(self._transitions) == self._transitions.maxlen

            # This is the typical case, yield a single n-step transition.
            yield build_n_step_transition_task(self._transitions)

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be FIRST."""
        self._transitions.clear()
        self._timestep_tm1 = None
        self._a_tm1 = None
        self._task = None



class NStepTransitionTaskPrevActionAccumulator:
    """Accumulates timesteps with task to form n-step transitions.
    Let `t` be the index of a timestep within an episode and `T` be the index of
    the final timestep within an episode. Then given the step type of the timestep
    passed into `step()` the accumulator will:
    *   `FIRST`: yield nothing.
    *   `MID`: if `t < n`, yield nothing, else yield one n-step transition
        `s_{t - n} -> s_t`.
    *   `LAST`: yield all transitions that end at `s_t = s_T` from up to n steps
        away, specifically `s_{T - min(n, T)} -> s_T, ..., s_{T - 1} -> s_T`.
        These are `min(n, T)`-step, ..., `1`-step transitions.
    """

    def __init__(self, n, num_actions: int):
        self._transitions = collections.deque(maxlen=n)  # Store 1-step transitions.
        self.reset()
        self._num_actions = num_actions

    def step(
        self,
        timestep_t: dm_env.TimeStep,
        a_t: Action,
        task: Task,
    ) -> Iterable[Transition_Task]:
        """Accumulates timestep and resulting action, yields transitions."""
        if timestep_t.first():
            self.reset()

        # There are no transitions on the first timestep.
        if self._timestep_tm1 is None:
            assert self._a_tm1 is None
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            self._a_tm2 = None
            self._task = task
            return  # Empty iterable.

        self._transitions.append(
            Transition_Task_Prev_Action(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                a_tm2=self._a_tm2,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
                task=self._task,
            )
        )

        self._timestep_tm1 = timestep_t
        self._a_tm2 = self._a_tm1
        self._a_tm1 = a_t
        self._time += 1

        # print statements for debugging
        print("time:", self._time)
        print("a_tm1", self._a_tm1)
        print("a_tm2", self._a_tm2)

        if timestep_t.last():
            # Yield any remaining n, n-1, ..., 1-step transitions at episode end.
            while self._transitions:
                yield build_n_step_transition_task_prev_action(self._transitions, self._a_tm2)
                self._transitions.popleft()
        else:
            # Wait for n transitions before yielding anything.
            if len(self._transitions) < self._transitions.maxlen:
                return  # Empty iterable.

            assert len(self._transitions) == self._transitions.maxlen

            # This is the typical case, yield a single n-step transition.
            yield build_n_step_transition_task_prev_action(self._transitions, self._a_tm2)

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be FIRST."""
        self._transitions.clear()
        self._timestep_tm1 = None
        self._a_tm1 = None
        self._a_tm2 = None
        self._task = None
        self._time = 0


class NStepTransitionTaskOneHotActionAccumulator:
    """Accumulates timesteps with task to form n-step transitions.
    Let `t` be the index of a timestep within an episode and `T` be the index of
    the final timestep within an episode. Then given the step type of the timestep
    passed into `step()` the accumulator will:
    *   `FIRST`: yield nothing.
    *   `MID`: if `t < n`, yield nothing, else yield one n-step transition
        `s_{t - n} -> s_t`.
    *   `LAST`: yield all transitions that end at `s_t = s_T` from up to n steps
        away, specifically `s_{T - min(n, T)} -> s_T, ..., s_{T - 1} -> s_T`.
        These are `min(n, T)`-step, ..., `1`-step transitions.
    """

    def __init__(self, n, num_actions: int):
        self._transitions = collections.deque(maxlen=n)  # Store 1-step transitions.
        self.reset()
        self._num_actions = num_actions
        self._a_tm1_vector = None

    def step(
        self,
        timestep_t: dm_env.TimeStep,
        a_t: Action,
        task: Task,
    ) -> Iterable[Transition_Task_OneHotAction]:
        """Accumulates timestep and resulting action, yields transitions."""
        if timestep_t.first():
            self.reset()

        # There are no transitions on the first timestep.
        if self._timestep_tm1 is None:
            assert self._a_tm1 is None
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            self._task = task
            self._a_tm1_vector = np.zeros(self._num_actions)
            self._a_tm1_vector[self._a_tm1] = 1
            return  # Empty iterable.

        self._transitions.append(
            Transition_Task_OneHotAction(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                a_tm1_vector=self._a_tm1_vector,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
                task=self._task,
            )
        )

        self._timestep_tm1 = timestep_t
        self._a_tm1 = a_t

        if timestep_t.last():
            # Yield any remaining n, n-1, ..., 1-step transitions at episode end.
            while self._transitions:
                yield build_n_step_transition_task_one_hot_action(self._transitions)
                self._transitions.popleft()
        else:
            # Wait for n transitions before yielding anything.
            if len(self._transitions) < self._transitions.maxlen:
                return  # Empty iterable.

            assert len(self._transitions) == self._transitions.maxlen

            # This is the typical case, yield a single n-step transition.
            yield build_n_step_transition_task_one_hot_action(self._transitions)

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be FIRST."""
        self._transitions.clear()
        self._timestep_tm1 = None
        self._a_tm1 = None
        self._task = None
        self._a_tm1_vector = None


class NStepTransitionSFAccumulator:
    """Accumulates timesteps to form n-step transitions.
    Let `t` be the index of a timestep within an episode and `T` be the index of
    the final timestep within an episode. Then given the step type of the timestep
    passed into `step()` the accumulator will:
    *   `FIRST`: yield nothing.
    *   `MID`: if `t < n`, yield nothing, else yield one n-step transition
        `s_{t - n} -> s_t`.
    *   `LAST`: yield all transitions that end at `s_t = s_T` from up to n steps
        away, specifically `s_{T - min(n, T)} -> s_T, ..., s_{T - 1} -> s_T`.
        These are `min(n, T)`-step, ..., `1`-step transitions.
    """

    def __init__(self, n: int, num_actions: int):
        self._transitions = collections.deque(maxlen=n)  # Store 1-step transitions.
        self.reset()
        self._num_actions = num_actions
        self._a_tm1_vector = None

    def step(
        self, timestep_t: dm_env.TimeStep, a_t: Action, task: Task
    ) -> Iterable[Transition]:
        """Accumulates timestep and resulting action, yields transitions."""
        if timestep_t.first():
            self.reset()

        # There are no transitions on the first timestep.
        if self._timestep_tm1 is None:
            assert self._a_tm1 is None
            if not timestep_t.first():
                raise ValueError("Expected FIRST timestep, got %s." % str(timestep_t))
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            self._task = task
            self._a_tm1_vector = np.zeros(self._num_actions)
            self._a_tm1_vector[self._a_tm1] = 1
            return  # Empty iterable.

        a_t_vector = np.zeros(self._num_actions)
        a_t_vector[a_t] = 1
        self._transitions.append(
            Transition_SF(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                a_tm1_vector=self._a_tm1_vector,
                r_t=timestep_t.reward,
                discount_t=timestep_t.discount,
                s_t=timestep_t.observation,
                a_t_vector=a_t_vector,
                task=self._task,
            )
        )

        self._timestep_tm1 = timestep_t
        self._a_tm1 = a_t

        if timestep_t.last():
            # Yield any remaining n, n-1, ..., 1-step transitions at episode end.
            while self._transitions:
                yield build_n_step_SF_transition(
                    self._transitions, self._num_actions, a_t
                )
                self._transitions.popleft()
        else:
            # Wait for n transitions before yielding anything.
            if len(self._transitions) < self._transitions.maxlen:
                return  # Empty iterable.

            assert len(self._transitions) == self._transitions.maxlen

            # This is the typical case, yield a single n-step transition.
            yield build_n_step_SF_transition(self._transitions, self._num_actions, a_t)

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be FIRST."""
        self._transitions.clear()
        self._timestep_tm1 = None
        self._a_tm1 = None
        self._a_tm1_vector = None
        self._task = None
