from flax import linen as nn
import jax.numpy as jnp
import jax
import chex

from .encoder import MinatarEncoder
from .helpers import normalize
from .output_types import SFNetOutputs,SFConsolidationNetOutputs


class SFConsolidationNetworkMinatar(nn.Module):
    """
    A simple consolidation network on the SFs.
    """

    num_actions: int
    feature_dim: int
    hidden_dim: int
    sf_dim: int
    num_beakers: int
    update_encoder_from_consolidation: bool = False

    def setup(self) -> None:
        self.encoder = {
            f"encoder_{i}": MinatarEncoder(name=f"MinatarEncoder_beaker{i}")
            for i in range(self.num_beakers)
        }

        self.rep_hidden = {
            f"rep_hidden_{i}": nn.Dense(self.sf_dim, name=f"rep_hidden_beaker{i}") for i in range(self.num_beakers)
        }

        self.fc_hidden = {
            f"fc_hidden_{i}": nn.Dense(self.feature_dim, name=f"fc_hidden_beaker{i}")
            for i in range(self.num_beakers)
        }

        # Store different layers for each beaker using dictionaries
        self.sf_consolidation_1 = {
            f"sf_consolidation1_{i}": nn.Dense(self.hidden_dim, name=f"sf_consolidation1_beaker{i}")
            for i in range(self.num_beakers)
        }

        self.sf_consolidation_2 = {
            f"sf_consolidation2_{i}": nn.Dense(self.sf_dim * self.num_actions, name=f"sf_consolidation2_beaker{i}")
            for i in range(self.num_beakers)
        }

    def sf_transform(self, features, idx):
        key_sf_consolidation1 = f"sf_consolidation1_{idx}"
        key_sf_consolidation2 = f"sf_consolidation2_{idx}"

        sf = self.sf_consolidation_1[key_sf_consolidation1](features)
        sf = nn.relu(sf)
        sf = self.sf_consolidation_2[key_sf_consolidation2](sf)
        return jnp.reshape(sf, (-1, self.num_actions, self.sf_dim))

    def __call__(self, obs, task):
        batch_size = obs.shape[0]

        encoder_list = [
            self.encoder[f"encoder_{i}"](obs) for i in range(self.num_beakers)
        ]

        encoder_all_beakers = jnp.stack(encoder_list, axis=1)

        # reshape encoder_all_beakers to (batch_size, num_beakers, -1)
        encoder_all_beakers = encoder_all_beakers.reshape(
            (batch_size, self.num_beakers, -1)
        )

        rep_hidden_list = [
            self.rep_hidden[f"rep_hidden_{i}"](encoder_all_beakers[:, i])
            for i in range(self.num_beakers)
        ]

        rep_hidden_all_beakers = jnp.stack(
            rep_hidden_list, axis=1
        )  # (batch_size, num_beakers, sf_dim)
        rep_hidden_all_beakers = nn.relu(rep_hidden_all_beakers)

        # basis features only depends on first beaker
        basis_features = normalize()(rep_hidden_all_beakers[:, 0, :])

        # normalize task and concatenate with representation
        task_normalized = normalize()(task)
        task_normalized = jnp.expand_dims(task_normalized, 1)
        task_normalized = jnp.tile(task_normalized, (1, self.num_beakers, 1))

        chex.assert_shape(
            task_normalized,
            (
                batch_size,
                self.num_beakers,
                self.sf_dim,
            ),
        )

        rep_task = jnp.concatenate(
            [rep_hidden_all_beakers, task_normalized], axis=2
        )  # (batch_size, num_beakers, sf_dim + task_dim)

        chex.assert_shape(
            rep_task,
            (
                batch_size,
                self.num_beakers,
                self.sf_dim + self.sf_dim,
            ),
        )

        # learn features using observation and task
        features_critic_sf_list = [
            self.fc_hidden[f"fc_hidden_{i}"](rep_task[:, i, :])
            for i in range(self.num_beakers)
        ]

        features_critic_sf_all_beakers = jnp.stack(
            features_critic_sf_list, axis=1
        )  # (batch_size, num_beakers, feature_dim)

        features_critic_sf_all_beakers = nn.relu(features_critic_sf_all_beakers)

        # learn sf using features
        sf_list_consolidation = [
            self.sf_transform(features_critic_sf_all_beakers[:, i, :], i)
            for i in range(self.num_beakers)
        ]

        sf_all_beakers = jnp.stack(
            sf_list_consolidation, axis=1
        )  # (batch_size, num_beakers, num_actions, sf_dim)

        chex.assert_shape(
            sf_all_beakers,
            (
                batch_size,
                self.num_beakers,
                self.num_actions,
                self.sf_dim,
            ),
        )

        """
        Compute Q-values. Make sure to use the SF without stop grad on the features. 
        """
        sf_first_beaker = sf_all_beakers[:, 0, :, :]

        chex.assert_shape(sf_first_beaker, (batch_size, self.num_actions, self.sf_dim))

        q_1 = jnp.einsum(
            "bi, bij -> bj", task, jnp.swapaxes(sf_first_beaker, 1, 2)
        ).reshape(-1, self.num_actions)

        return SFConsolidationNetOutputs(
            basis_features=basis_features,
            sf=jnp.expand_dims(sf_all_beakers[:, 0, :, :], axis=1),
            # to be used for consolidation loss
            q_1=q_1,
            sf_consolidation=sf_all_beakers[:, 1:, :, :],
        )


class SFConsolidationIndividualNetworkMinatar(nn.Module):
    """
    A simple SF-network for Minatar environments based on our NeurIPS 2024 paper,
    "Learning Successor Features the Simple Way".
    """

    num_actions: int
    feature_dim: int
    hidden_dim: int
    sf_dim: int
    num_beakers: int
    update_encoder_from_consolidation: bool = False

    def setup(self) -> None:
        self.encoder = MinatarEncoder()
        self.rep_hidden = nn.Dense(features=self.sf_dim)
        self.fc_hidden = nn.Dense(features=self.feature_dim)
        self.fc_sf = nn.Dense(features=self.sf_dim * self.num_actions)

    def __call__(self, obs, task):
        rep = self.encoder(obs)
        rep = rep.reshape((rep.shape[0], -1))

        rep_hidden = self.rep_hidden(rep)
        rep_hidden = nn.relu(rep_hidden)
        basis_features = normalize()(rep_hidden)

        # normalize task and concatenate with representation
        task_normalized = normalize()(task)
        rep_task = jnp.concatenate([rep_hidden, task_normalized], axis=1)

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
