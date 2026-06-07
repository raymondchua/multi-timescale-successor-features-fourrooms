import flax.linen as nn
from .encoder import MinatarEncoder, CNN, DQNEncoder
from .output_types import QNetOutputs
from helpers import BatchRenorm
import jax.numpy as jnp


class QNetworkMinatar(nn.Module):
    """
    A simple Q-network for Minatar environments.
    """

    feature_dim: int
    num_actions: int

    def setup(self) -> None:
        self.encoder = MinatarEncoder()
        self.fc_hidden = nn.Dense(features=self.feature_dim)
        self.output = nn.Dense(features=self.num_actions)

    def __call__(self, x):
        rep = self.encoder(x)
        x = rep.reshape((rep.shape[0], -1))
        x = self.fc_hidden(x)
        x = nn.relu(x)
        x = self.output(x)
        return QNetOutputs(
            q_1=x,
            obs_rep=rep,
        )


class DQNetwork(nn.Module):
    """
    DQN architecture.
    """

    feature_dim: int
    hidden_dim: int
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = DQNEncoder()(x)
        x = nn.relu(x)
        rep = nn.Dense(features=self.feature_dim)(x)
        x = nn.relu(rep)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_actions)(x)
        return QNetOutputs(q_1=x, obs_rep=rep)


class PQNetwork_atari(nn.Module):
    """
    Following PQN github repo: https://github.com/mttga/purejaxql
    """

    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.action_dim)(x)
        return x


class PQNetwork_craftax(nn.Module):
    """
    Following PQN github repo: https://github.com/mttga/purejaxql
    """

    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "batch_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)

        return x


class DQNetwork_named(nn.Module):
    feature_dim: int
    hidden_dim: int
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, return_hs: bool = False):
        x = DQNEncoder(name="encoder")(x)
        x = nn.relu(x)

        rep_pre = nn.Dense(self.feature_dim, name="rep")(x)
        rep = nn.relu(rep_pre)

        hid_pre = nn.Dense(self.hidden_dim, name="hidden")(rep)
        hid = nn.relu(hid_pre)

        q = nn.Dense(self.num_actions, name="q")(hid)

        out = QNetOutputs(q_1=q, obs_rep=rep)
        if return_hs:
            # CBP will use these hidden activations (post-nonlinearity)
            return out, [rep, hid]
        return out
