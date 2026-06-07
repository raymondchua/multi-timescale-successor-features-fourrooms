import optax
from typing import Literal, Optional


def get_optimizer(
    optim_type: Literal["adam", "rmsprop", "sgd"],
    lr: float,
    b1: Optional[float] = 0.9,
    b2: Optional[float] = 0.999,
) -> optax.GradientTransformation:
    if optim_type == "adam":
        # Use optax.adam, but tell optax that we'd like to move the adam hyperparameters into the optimizer's state
        # so that we can log them.
        optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=lr, b1=b1, b2=b2, eps=0.00015
        )
        # optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2, eps=0.00015)

    elif optim_type == "rmsprop":
        optimizer = optax.rmsprop(
            learning_rate=lr,
            decay=0.95,
            eps=0.01 / 32**2,
            centered=True,
        )

    else:
        assert optim_type == "sgd"
        optimizer = optax.sgd(learning_rate=lr)
    return optimizer
