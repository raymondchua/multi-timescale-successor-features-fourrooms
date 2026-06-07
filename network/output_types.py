import typing
import jax.numpy as jnp

class FeatureNetOutputs(typing.NamedTuple):
    features: jnp.ndarray
    basis_features: jnp.ndarray

class QNetOutputs(typing.NamedTuple):
    obs_rep: jnp.ndarray
    q_1: jnp.ndarray

class QNetBeakersOutputs(typing.NamedTuple):
    q_1: jnp.ndarray
    q_current_beaker_0_stop_grad_feature: jnp.ndarray
    q_vals_consolidation: jnp.ndarray

class SFNetOutputs(typing.NamedTuple):
    basis_features: jnp.ndarray
    sf: jnp.ndarray
    q_1: jnp.ndarray

class SFNetRepOutputs(typing.NamedTuple):
    basis_features: jnp.ndarray
    sf: jnp.ndarray
    q_1: jnp.ndarray
    rep_hidden: jnp.ndarray

class SFNetOutputsWithoutBasis(typing.NamedTuple):
    sf: jnp.ndarray
    q_1: jnp.ndarray

class SFRNNNetOutputs(typing.NamedTuple):
    hidden: list
    basis_features: jnp.ndarray
    sf: jnp.ndarray
    q_1: jnp.ndarray

class SFReconstructionNetOutputs(typing.NamedTuple):
    basis_features: jnp.ndarray
    sf: jnp.ndarray
    q_1: jnp.ndarray
    s_t: jnp.ndarray

class SFLSTMNetOutputs(typing.NamedTuple):
    basis_features: jnp.ndarray
    sf: jnp.ndarray
    q_1: jnp.ndarray
    hidden_state: list

class SFConsolidationNetOutputs(typing.NamedTuple):
    basis_features: jnp.ndarray
    sf: jnp.ndarray
    q_1: jnp.ndarray
    sf_consolidation: jnp.ndarray

class SFConsolidationAttentionNetOutputs(typing.NamedTuple):
    basis_features: jnp.ndarray
    sf: jnp.ndarray
    q_1: jnp.ndarray
    sf_consolidation: jnp.ndarray
    attention_logits: jnp.ndarray
    attention_outputs: jnp.ndarray
    attended_sf: jnp.ndarray
    sf_before_mask: jnp.ndarray
    sf_after_mask: jnp.ndarray
    keys: jnp.ndarray
    values: jnp.ndarray


class SFAttentionNetOutputs(typing.NamedTuple):
    attention_logits: jnp.ndarray
    attention_outputs: jnp.ndarray
    attended_sf: jnp.ndarray
    q_1: jnp.ndarray
    keys: jnp.ndarray
    values: jnp.ndarray
