import chex
from flax.linen.normalization import compact

from .encoder import MinatarEncoder
from .helpers import normalize
from .output_types import SFAttentionNetOutputs, SFConsolidationAttentionNetOutputs

import jax
import flax.linen as nn
import jax.numpy as jnp


class SFSoftmaxAttentionNetwork(nn.Module):
    """
    Softmax attention mechanism on the SF. Inputs are the SFs from all beakers,
    and the task. The values for the attention mechanism are each set of SFs, which are the addition of the first SF and
    the difference between the first SFs and the SFs of the current beaker. The Q-values are then computed based on the
    attended SF.
    """

    num_actions: int
    sf_dim: int
    num_beakers: int
    apply_mask_to_keys: bool = False,
    apply_gain_sf_diff: bool = False,
    predefined_gain: bool = False,

    def setup(self) -> None:
        self.keys_dim = self.sf_dim * 2
        self.query = nn.Dense(features=self.sf_dim, use_bias=False)
        self.keys = nn.Dense(features=self.sf_dim, use_bias=False)

        if self.apply_gain_sf_diff:
            self.sf_keys_gain = self.param(
                "sf_keys_gain", nn.initializers.xavier_uniform(), (self.num_beakers - 1, self.num_actions)
            )

            self.sf_values_gain = self.param(
                "sf_values_gain", nn.initializers.xavier_uniform(), (self.num_beakers - 1, self.num_actions)
            )

    def __call__(self, sf_beakers, task, mask, recall_gain):
        B, N, A, D = sf_beakers.shape
        assert N == self.num_beakers, f"Number of beakers should be {self.num_beakers}, but got {N}"
        assert A == self.num_actions, f"Number of actions should be {self.num_actions}, but got {A}"
        assert D == self.sf_dim, f"SF dimension should be {self.sf_dim}, but got {D}"

        # normalize task and concatenate with representation
        task_expanded = jnp.expand_dims(task, 1)    # (batch_size, 1, sf_dim)
        query = self.query(task_expanded)

        # Add sf from the first beaker to all other beakers
        sf_beakers_keys_diff = jnp.expand_dims(sf_beakers[:, 0], axis=1) - sf_beakers
        sf_beakers_values_diff = jnp.expand_dims(sf_beakers[:, 0], axis=1) - sf_beakers

        if self.apply_gain_sf_diff:
            sf_keys_gain_tiled = jnp.expand_dims(self.sf_keys_gain, axis=(0, 3))
            sf_keys_gain_tiled = jnp.tile(sf_keys_gain_tiled, (1, 1, 1, self.sf_dim))
            sf_beakers_keys_diff = sf_beakers_keys_diff.at[:, 1:].set(
                sf_beakers_keys_diff[:, 1:] * sf_keys_gain_tiled)

            sf_values_gain_tiled = jnp.expand_dims(self.sf_values_gain, axis=(0, 3))
            sf_values_gain_tiled = jnp.tile(sf_values_gain_tiled, (1, 1, 1, self.sf_dim))
            sf_beakers_values_diff = sf_beakers_values_diff.at[:, 1:].set(
                sf_beakers_values_diff[:, 1:] * sf_values_gain_tiled)

        elif self.predefined_gain:
            sf_keys_gain_tiled = jnp.expand_dims(recall_gain, axis=(0, 3))
            sf_keys_gain_tiled = jnp.tile(sf_keys_gain_tiled, (1, 1, 1, self.sf_dim))
            sf_beakers_keys_diff = sf_beakers_keys_diff.at[:, 1:].set(
                sf_beakers_keys_diff[:, 1:] * sf_keys_gain_tiled)

            sf_values_gain_tiled = jnp.expand_dims(recall_gain, axis=(0, 3))
            sf_values_gain_tiled = jnp.tile(sf_values_gain_tiled, (1, 1, 1, self.sf_dim))
            sf_beakers_values_diff = sf_beakers_values_diff.at[:, 1:].set(
                sf_beakers_values_diff[:, 1:] * sf_values_gain_tiled)

        sf_first_beaker = jnp.expand_dims(sf_beakers[:, 0], axis=1)
        sf_first_beaker = jnp.tile(sf_first_beaker, (1, self.num_beakers, 1, 1))

        sf_beakers_keys = sf_first_beaker + sf_beakers_keys_diff
        sf_beakers_values = sf_first_beaker + sf_beakers_values_diff

        keys = self.keys(sf_beakers_keys)

        # shape: (batch_size, 1, num_beakers, num_actions)
        attn_logits = jnp.einsum("bqf,bnaf->bqna", query, keys) / jnp.sqrt(self.sf_dim)

        if self.apply_mask_to_keys:
            mask = jnp.reshape(mask, (1, 1, mask.shape[0], 1))  # Reshape to (1, 1, 5, 1)
            mask = jnp.tile(mask, (1, 1, 1, attn_logits.shape[-1]))  # Tile to (1, 1, 5, 4)
            attn_logits = jnp.where(mask > 0, attn_logits, -1e9 * jnp.ones_like(attn_logits))

        attention_weights = jax.nn.softmax(attn_logits, axis=2)

        attended_sf = jnp.einsum("bqna,bnaf->bqaf", attention_weights, sf_beakers_values)

        attended_sf = attended_sf.squeeze(1).swapaxes(1, 2)

        # Compute Q-values
        q_1 = jnp.einsum("bi,bij->bj", task, attended_sf)

        return SFAttentionNetOutputs(
            q_1=q_1,
            attention_logits=attn_logits,
            attention_outputs=attention_weights,
            attended_sf=attended_sf,
            keys=keys,
            values=sf_beakers_values,
        )


class SFSoftmaxAttentionDiffNetwork(nn.Module):
    """
    Softmax attention mechanism on the difference of the SF beakers. Inputs are the SFs from all beakers,
    and the task. The values for the attention mechanism are each set of SFs, which are the addition of the first SF and
    the difference between the first SFs and the SFs of the current beaker. The Q-values are then computed based on the
    attended SF.
    """

    num_actions: int
    sf_dim: int
    num_beakers: int
    apply_mask_to_keys: bool = False,
    apply_gain_sf_diff: bool = False,
    layer_norm_keys: bool = False,
    layer_norm_values: bool = False,
    learnable_layer_norm_parameters: bool = False,

    def setup(self) -> None:
        self.keys_dim = self.sf_dim * 2
        self.query = nn.Dense(features=self.sf_dim, use_bias=False)
        self.keys = nn.Dense(features=self.sf_dim, use_bias=False)

        if self.apply_gain_sf_diff:
            self.sf_values_gain = self.param(
                "sf_values_gain", nn.initializers.ones, (1, )
            )

        if self.layer_norm_keys:
            self.keys_layer_norm = nn.LayerNorm(
                use_scale=self.learnable_layer_norm_parameters, use_bias=self.learnable_layer_norm_parameters
            )

        if self.layer_norm_values:
            self.values_layer_norm = nn.LayerNorm(
                use_scale=self.learnable_layer_norm_parameters, use_bias=self.learnable_layer_norm_parameters
            )

    def __call__(self, sf_beakers, task, mask, recall_gain):
        B, N, A, D = sf_beakers.shape
        assert N == self.num_beakers, f"Number of beakers should be {self.num_beakers}, but got {N}"
        assert A == self.num_actions, f"Number of actions should be {self.num_actions}, but got {A}"
        assert D == self.sf_dim, f"SF dimension should be {self.sf_dim}, but got {D}"

        # normalize task and concatenate with representation
        task_expanded = jnp.expand_dims(task, 1)    # (batch_size, 1, sf_dim)
        query = self.query(task_expanded)

        # Subtract all other beakers from the first beaker
        sf_first_beaker = jnp.expand_dims(sf_beakers[:, 0], 1)
        sf_first_beaker_tiled = jnp.tile(sf_first_beaker, (1, self.num_beakers, 1, 1))
        sf_beakers_keys_diff = sf_first_beaker_tiled - sf_beakers
        sf_beakers_values_diff = sf_first_beaker_tiled - sf_beakers

        sf_beakers_keys = sf_beakers_keys_diff[:, 1:] # [B, N - 1, A, D]
        sf_beakers_values = sf_beakers_values_diff[:, 1:] # [B, N - 1, A, D]

        if self.layer_norm_keys:
            sf_beakers_keys_reshape = jnp.swapaxes(sf_beakers_keys, 1, 2) # [B, A, N - 1, D]
            keys1 = jnp.reshape(sf_beakers_keys_reshape, (B, A, -1))
            keys1 = self.keys_layer_norm(keys1)

            # reshape back to [B, A, N - 1, sf_dim]
            keys1 = jnp.reshape(keys1, (B, A, self.num_beakers - 1, self.sf_dim))
            sf_beakers_keys = jnp.swapaxes(keys1, 1, 2)  # [B, N - 1, A, D]

        if self.layer_norm_values:
            sf_beakers_values_diff_reshape = jnp.swapaxes(sf_beakers_values, 1, 2)  # [B, A, N - 1, D]
            values1 = jnp.reshape(sf_beakers_values_diff_reshape, (B, A, -1))
            values1 = self.values_layer_norm(values1)

            # reshape back to [B, N - 1, sf_dim]
            values1 = jnp.reshape(values1, (B, A, self.num_beakers - 1, self.sf_dim))
            sf_beakers_values = jnp.swapaxes(values1, 1, 2)  # [B, N - 1, A, D]

        keys = self.keys(sf_beakers_keys)

        # shape: (batch_size, 1, num_beakers, num_actions)
        attn_logits = jnp.einsum("bqf,bnaf->bqna", query, keys) / jnp.sqrt(self.sf_dim)

        attention_weights = jax.nn.softmax(attn_logits, axis=2)

        if self.apply_mask_to_keys:
            mask = mask[1:]  # Remove the first beaker mask
            mask = jnp.reshape(mask, (1, 1, mask.shape[0], 1))  # Reshape to (1, 1, N - 1 , 1)
            mask = jnp.tile(mask, (1, 1, 1, attention_weights.shape[-1]))  # Tile to (1, 1, N - 1, 4)
            attention_weights = jnp.where(mask > 0, attention_weights, jnp.zeros_like(attention_weights))

        attended_sf = jnp.einsum("bqna,bnaf->bqaf", attention_weights, sf_beakers_values)

        if self.apply_gain_sf_diff:
            attended_sf_diff = self.sf_values_gain * recall_gain[0] * attended_sf
            attended_sf = sf_beakers[:, 0]  +  attended_sf_diff.squeeze(1)
        else:
            attended_sf_diff = recall_gain[0] * attended_sf
            attended_sf = sf_beakers[:, 0]  + attended_sf_diff.squeeze(1)

        attended_sf = attended_sf.swapaxes(1,  2)

        # Compute Q-values
        q_1 = jnp.einsum("bi,bij->bj", task, attended_sf)

        return SFAttentionNetOutputs(
            q_1=q_1,
            attention_logits=attn_logits,
            attention_outputs=attention_weights,
            attended_sf=attended_sf,
            keys=keys,
            values=sf_beakers_values,
        )


class SFSoftmaxAttentionDiffUniqueNetwork(nn.Module):
    """
    Softmax attention mechanism on the difference of the SF beakers. Inputs are the SFs from all beakers,
    and the task. The values for the attention mechanism are each set of SFs, which are the addition of the first SF and
    the difference between the first SFs and the SFs of the current beaker. The Q-values are then computed based on the
    attended SF.
    """

    num_actions: int
    sf_dim: int
    num_beakers: int
    apply_mask_to_keys: bool = False,
    apply_gain_sf_diff: bool = False,
    layer_norm_keys: bool = False,
    layer_norm_values: bool = False,
    learnable_layer_norm_parameters: bool = False,

    def setup(self) -> None:
        self.keys_dim = self.sf_dim * 2
        self.query = nn.Dense(features=self.sf_dim, use_bias=False)
        self.keys = nn.Dense(features=self.sf_dim, use_bias=False)

        if self.apply_gain_sf_diff:
            self.sf_values_gain = self.param(
                "sf_values_gain", nn.initializers.ones, (1, )
            )

        if self.layer_norm_keys:
            self.keys_layer_norm = nn.LayerNorm(
                use_scale=self.learnable_layer_norm_parameters, use_bias=self.learnable_layer_norm_parameters
            )

        if self.layer_norm_values:
            self.values_layer_norm = nn.LayerNorm(
                use_scale=self.learnable_layer_norm_parameters, use_bias=self.learnable_layer_norm_parameters
            )

    def __call__(self, sf_beakers, task, mask, recall_gain):
        B, N, A, D = sf_beakers.shape
        assert N == self.num_beakers, f"Number of beakers should be {self.num_beakers}, but got {N}"
        assert A == self.num_actions, f"Number of actions should be {self.num_actions}, but got {A}"
        assert D == self.sf_dim, f"SF dimension should be {self.sf_dim}, but got {D}"

        # normalize task and concatenate with representation
        task_expanded = jnp.expand_dims(task, 1)    # (batch_size, 1, sf_dim)
        query = self.query(task_expanded)

        # Subtract all other beakers from the its previous beaker
        sf_beakers_keys = sf_beakers[:, 1:] - sf_beakers[:, :-1]
        sf_beakers_values = sf_beakers[:, 1:] - sf_beakers[:, :-1]

        if self.layer_norm_keys:
            sf_beakers_keys_reshape = jnp.swapaxes(sf_beakers_keys, 1, 2) # [B, A, N - 1, D]
            keys1 = jnp.reshape(sf_beakers_keys_reshape, (B, A, -1))
            keys1 = self.keys_layer_norm(keys1)

            # reshape back to [B, A, N - 1, sf_dim]
            keys1 = jnp.reshape(keys1, (B, A, self.num_beakers - 1, self.sf_dim))
            sf_beakers_keys = jnp.swapaxes(keys1, 1, 2)  # [B, N - 1, A, D]

        if self.layer_norm_values:
            sf_beakers_values_diff_reshape = jnp.swapaxes(sf_beakers_values, 1, 2)  # [B, A, N - 1, D]
            values1 = jnp.reshape(sf_beakers_values_diff_reshape, (B, A, -1))
            values1 = self.values_layer_norm(values1)

            # reshape back to [B, N - 1, sf_dim]
            values1 = jnp.reshape(values1, (B, A, self.num_beakers - 1, self.sf_dim))
            sf_beakers_values = jnp.swapaxes(values1, 1, 2)  # [B, N - 1, A, D]

        keys = self.keys(sf_beakers_keys)

        # shape: (batch_size, 1, num_beakers, num_actions)
        attn_logits = jnp.einsum("bqf,bnaf->bqna", query, keys) / jnp.sqrt(self.sf_dim)

        attention_weights = jax.nn.softmax(attn_logits, axis=2)

        if self.apply_mask_to_keys:
            mask = mask[1:]  # Remove the first beaker mask
            mask = jnp.reshape(mask, (1, 1, mask.shape[0], 1))  # Reshape to (1, 1, N - 1 , 1)
            mask = jnp.tile(mask, (1, 1, 1, attention_weights.shape[-1]))  # Tile to (1, 1, N - 1, 4)
            attention_weights = jnp.where(mask > 0, attention_weights, jnp.zeros_like(attention_weights))

        attended_sf = jnp.einsum("bqna,bnaf->bqaf", attention_weights, sf_beakers_values)

        if self.apply_gain_sf_diff:
            attended_sf_diff = self.sf_values_gain * recall_gain[0] * attended_sf
            attended_sf = sf_beakers[:, 0]  +  attended_sf_diff.squeeze(1)
        else:
            attended_sf_diff = recall_gain[0] * attended_sf
            attended_sf = sf_beakers[:, 0]  + attended_sf_diff.squeeze(1)

        attended_sf = attended_sf.swapaxes(1,  2)

        # Compute Q-values
        q_1 = jnp.einsum("bi,bij->bj", task, attended_sf)

        return SFAttentionNetOutputs(
            q_1=q_1,
            attention_logits=attn_logits,
            attention_outputs=attention_weights,
            attended_sf=attended_sf,
            keys=keys,
            values=sf_beakers_values,
        )