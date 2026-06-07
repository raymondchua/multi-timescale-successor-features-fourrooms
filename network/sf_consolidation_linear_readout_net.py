from flax.linen.normalization import compact

from .encoder import MinatarEncoder
from .helpers import normalize
from .output_types import SFConsolidationNetOutputs, SFNetOutputsWithoutBasis

import jax
import flax.linen as nn
import jax.numpy as jnp


class SFConsolidationLinearReadoutNetworkMinatar(nn.Module):
    """
    Add synaptic consolidation to the SF network and use a linear readout.
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
        self.linear_readout = nn.Dense(features=self.sf_dim)

        # Store different layers for each beaker using dictionaries
        self.sf_consolidation_1 = {
            f"sf_consolidation1_{i}": nn.Dense(self.hidden_dim)
            for i in range(self.num_beakers)
        }

        self.sf_consolidation_2 = {
            f"sf_consolidation2_{i}": nn.Dense(self.sf_dim * self.num_actions)
            for i in range(self.num_beakers)
        }

    def __call__(self, obs, task):
        batch_size = obs.shape[0]

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

        sf_list = []
        for i in range(self.num_beakers):
            if i != 0:
                features_critic_sf = jax.lax.stop_gradient(
                    features_critic_sf
                )  # ensure only the first beaker is trainable

            # Retrieve layers dynamically from the dictionary
            sf_dense_1 = self.sf_consolidation_1[f"sf_consolidation1_{i}"]
            sf_current_beaker = sf_dense_1(features_critic_sf)
            sf_current_beaker = nn.relu(sf_current_beaker)

            sf_dense_2 = self.sf_consolidation_2[f"sf_consolidation2_{i}"]
            sf_current_beaker = sf_dense_2(sf_current_beaker)

            sf_current_beaker_reshape = jnp.reshape(
                sf_current_beaker, (-1, self.num_actions, self.sf_dim)
            )
            sf_list.append(sf_current_beaker_reshape)

        sf_stacked = jnp.stack(
            sf_list, axis=1
        )  # (batch_size, num_beakers, num_actions, sf_dim)

        assert sf_stacked.shape == (
            batch_size,
            self.num_beakers,
            self.num_actions,
            self.sf_dim,
        )

        sf_readout_list = []

        for i in range(self.num_actions):
            sf_all_beakers_current_action = sf_stacked[
                                            :, :, i, :
                                            ]  # shape: [batch_size, num_beakers, sf_dim]

            # flatten sf_all_beakers_current_action so that the shape becomes [batch_size, num_beakers*sf_dim]
            sf_all_beakers_current_action_flattened = jnp.reshape(
                sf_all_beakers_current_action, (-1, self.num_beakers * self.sf_dim)
            )

            # Use a linear layer to attend to all the beakers
            sf_action_a = self.linear_readout(sf_all_beakers_current_action_flattened) # shape: [batch_size, sf_dim]

            sf_readout_list.append(sf_action_a)

        # convert attended_sf from list to jnp array
        sf_readout_array = jnp.asarray(
            sf_readout_list
        )  # shape: [num_actions, batch_size, sf_dim]

        # reshape such that the final shape is [batch_size, sf_dim, num_actions]
        sf_readout_array = jnp.swapaxes(
            sf_readout_array, 0, 1
        )  # shape: [batch_size, num_actions, sf_dim]
        sf_readout_array = jnp.swapaxes(
            sf_readout_array, 1, 2
        )  # shape: [batch_size, sf_dim, num_actions]

        q_1 = jnp.einsum(
            "bi,bij->bj", task, sf_readout_array
        )  # shape: [batch_size, num_actions]

        sf_1 = sf_stacked[:, 0, :, :]  # shape: [batch_size, num_actions, sf_dim]
        sf_1 = jnp.expand_dims(sf_1, axis=1)   # shape: [batch_size, 1, num_actions, sf_dim]

        sf_consolidation = sf_stacked[: , 1:, :, :]  # shape: [batch_size, num_beakers-1, num_actions, sf_dim]

        return SFConsolidationNetOutputs(
            basis_features=basis_features,
            sf=sf_1,
            q_1=q_1,
            sf_consolidation=sf_consolidation,
        )

class SFLinearReadoutDiffNetwork(nn.Module):
    """
    Intead of using cross attention, we use a linear readout of the difference
    between the first beaker and all other beakers.
    """

    num_actions: int
    sf_dim: int
    num_beakers: int
    apply_mask_to_keys: bool = False,
    apply_gain_sf_diff: bool = False,

    def setup(self) -> None:
        self.linear_readout = nn.Dense(features=self.sf_dim)

        if self.apply_gain_sf_diff:
            self.sf_values_gain = self.param(
                "sf_values_gain", nn.initializers.ones, (1,)
            )

    def __call__(self, sf_beakers, task, mask, recall_gain):
        B, N, A, D = sf_beakers.shape
        assert N == self.num_beakers, f"Number of beakers should be {self.num_beakers}, but got {N}"
        assert A == self.num_actions, f"Number of actions should be {self.num_actions}, but got {A}"
        assert D == self.sf_dim, f"SF dimension should be {self.sf_dim}, but got {D}"

        # Subtract all other beakers from the first beaker
        sf_first_beaker = jnp.expand_dims(sf_beakers[:, 0], 1)
        sf_first_beaker_tiled = jnp.tile(sf_first_beaker, (1, self.num_beakers, 1, 1))
        sf_beakers_keys_diff = sf_first_beaker_tiled - sf_beakers

        sf_beakers_diff = sf_beakers_keys_diff[:, 1:]

        if self.apply_mask_to_keys:
            mask = mask[1:]  # Remove the first beaker mask
            mask = jnp.reshape(mask, (1, mask.shape[0], 1, 1))  # Reshape to (1, 1, N - 1, 1)
            mask = jnp.tile(mask, (B, 1, A, D))  # Tile to (B, N - 1, A, D)
            sf_beakers_diff = jnp.where(mask > 0, sf_beakers_diff, jnp.zeros_like(sf_beakers_diff))

        sf_beakers_diff = jnp.swapaxes(sf_beakers_diff, 1, 2)  # Swap axes to (B, A, N - 1, D)
        sf_beakers_diff = jnp.reshape(sf_beakers_diff, (B, A, (N-1) * D))
        sf_beakers_diff = self.linear_readout(sf_beakers_diff) # Shape: (B, A, D)

        if self.apply_gain_sf_diff:
            scale = self.sf_values_gain * recall_gain[0]
        else:
            scale = recall_gain[0]
        scale = jnp.reshape(scale, (1, -1, 1))  # Reshape to (1, num_actions, 1)
        scale = jnp.tile(scale, (B, 1, D))  # Tile to (B, num_actions, sf_dim)

        sf_diff_readout = scale * sf_beakers_diff # Shape: (B, A, D)
        sf_diff_readout = sf_beakers[:, 0] + sf_diff_readout

        sf_diff_readout = jnp.swapaxes(sf_diff_readout, 1, 2)  # Shape: (B, D, A)

        # Compute Q-values
        q_1 = jnp.einsum("bi,bij->bj", task, sf_diff_readout)

        return SFNetOutputsWithoutBasis(
            sf=sf_diff_readout,
            q_1=q_1,
        )


