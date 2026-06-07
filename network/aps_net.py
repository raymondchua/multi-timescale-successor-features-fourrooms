import jax.lax

from .encoder import DQNEncoder
from .helpers import normalize
from .output_types import SFNetOutputs, SFNetRepOutputs

import flax.linen as nn
import jax.numpy as jnp

class APSNetwork(nn.Module):
    """
    A network that use pre-training to learn the basis features and successor features based on
    the APS: Active Pretraining with Successor Features paper.
    https://arxiv.org/abs/2108.13956
    """

    hidden_dim: int
    num_actions: int
    sf_dim: int

    def setup(self) -> None:
        self.encoder = DQNEncoder()
        self.rep_hidden = nn.Dense(features=self.sf_dim)
        self.fc_hidden = nn.Dense(features=self.hidden_dim)
        self.fc_sf = nn.Dense(features=self.sf_dim * self.num_actions)

    def __call__(self, obs, task):
        rep = self.encoder(obs)
        rep = rep.reshape((rep.shape[0], -1))

        rep_hidden = self.rep_hidden(rep)
        rep_hidden = nn.relu(rep_hidden)
        basis_features = normalize()(rep_hidden)

        # concatenate task with representation
        rep_task = jnp.concatenate([rep_hidden, task], axis=1)

        # features for SF
        features_critic_sf = self.fc_hidden(rep_task)
        features_critic_sf = nn.relu(features_critic_sf)

        # SF
        sf = self.fc_sf(features_critic_sf)
        sf_action = jnp.reshape(
            sf,
            (
                -1,
                self.sf_dim,
                self.num_actions,
            ),
        )  # (batch_size, sf_dim, num_actions)

        q_1 = jnp.einsum("bi, bij -> bj", task, sf_action).reshape(
            -1, self.num_actions
        )  # (batch_size, num_actions)

        return SFNetOutputs(
            basis_features=basis_features,
            sf=sf_action,
            q_1=q_1,
        )

class APSL1Network(nn.Module):
    """
    A network that use pre-training to learn the basis features and successor features based on
    the APS: Active Pretraining with Successor Features paper.
    https://arxiv.org/abs/2108.13956
    Added an output to induce sparsity on inputs of the basis features
    """

    hidden_dim: int
    num_actions: int
    sf_dim: int

    def setup(self) -> None:
        self.encoder = DQNEncoder()
        self.rep_hidden = nn.Dense(features=self.sf_dim)
        self.fc_hidden = nn.Dense(features=self.hidden_dim)
        self.fc_sf = nn.Dense(features=self.sf_dim * self.num_actions)

    def __call__(self, obs, task):
        rep = self.encoder(obs)
        rep = rep.reshape((rep.shape[0], -1))

        rep_hidden = self.rep_hidden(rep)
        rep_hidden = nn.relu(rep_hidden)
        basis_features = normalize()(rep_hidden)

        # concatenate task with representation
        rep_task = jnp.concatenate([rep_hidden, task], axis=1)

        # features for SF
        features_critic_sf = self.fc_hidden(rep_task)
        features_critic_sf = nn.relu(features_critic_sf)

        # SF
        sf = self.fc_sf(features_critic_sf)
        sf_action = jnp.reshape(
            sf,
            (
                -1,
                self.sf_dim,
                self.num_actions,
            ),
        )  # (batch_size, sf_dim, num_actions)

        q_1 = jnp.einsum("bi, bij -> bj", task, sf_action).reshape(
            -1, self.num_actions
        )  # (batch_size, num_actions)

        return SFNetRepOutputs(
            basis_features=basis_features,
            sf=sf_action,
            q_1=q_1,
            rep_hidden=rep_hidden,
        )