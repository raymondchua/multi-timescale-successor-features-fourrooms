import jax
import jax.numpy as jnp
import flax.struct
from typing import Any, Callable, Optional


class EWCState(flax.struct.PyTreeNode):
    params_star: Any
    fisher: Any


def compute_fisher_diagonal(loss_fn: Callable, params: Any, data: Any) -> Any:
    """Compute diagonal Fisher information as squared gradients."""
    _, grads = jax.value_and_grad(loss_fn)(params, data)
    return jax.tree_map(lambda g: jnp.square(g), grads)


def update_fisher(old_fisher: Any, new_fisher: Any, gamma: float) -> Any:
    return jax.tree_map(
        lambda f_old, f_new: gamma * f_old + f_new, old_fisher, new_fisher
    )


def consolidate_params(params: Any) -> Any:
    return jax.tree_map(lambda x: x.copy(), params)


def ewc_penalty(
    params: Any, params_star: Any, fisher: Any, ewc_lambda: float
) -> jnp.ndarray:
    diffs = jax.tree_map(lambda p, p_star: p - p_star, params, params_star)
    penalties = jax.tree_map(lambda d, f: f * jnp.square(d), diffs, fisher)
    total = sum(jax.tree_util.tree_leaves(penalties))
    return 0.5 * ewc_lambda * total


def create_ewc_loss_fn(
    task_loss_fn: Callable, ewc_state: Optional[EWCState], ewc_lambda: float
) -> Callable:
    def loss_fn(params: Any, *args, **kwargs) -> jnp.ndarray:
        task_loss = task_loss_fn(params, *args, **kwargs)
        if ewc_state is not None and ewc_state.fisher is not None:
            reg = ewc_penalty(
                params, ewc_state.params_star, ewc_state.fisher, ewc_lambda
            )
            return task_loss + reg
        else:
            return task_loss

    return loss_fn
