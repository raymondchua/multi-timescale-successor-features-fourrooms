from flax import linen as nn
import jax
import jax.numpy as jnp
from .encoder import MinatarEncoder, CNN, DQNEncoder
from .output_types import QNetOutputs


class DQNetworkPI(nn.Module):
    feature_dim: int
    hidden_dim: int
    num_actions: int

    def setup(self):
        # stable submodule names come from these attribute names
        self.encoder = DQNEncoder()
        self.rep = nn.Dense(self.feature_dim)  # was Dense_0
        self.hidden = nn.Dense(self.hidden_dim)  # was Dense_1

        # Three heads from the same hidden
        self.head_old = nn.Dense(self.num_actions)
        self.head_new = nn.Dense(self.num_actions)
        self.head_copy = nn.Dense(self.num_actions)

    def __call__(self, x: jnp.ndarray, inject: bool = False):
        x = self.encoder(x)
        x = nn.relu(x)
        rep = self.rep(x)
        x = nn.relu(rep)
        x = self.hidden(x)
        x = nn.relu(x)

        q_old = self.head_old(x)
        q_new = self.head_new(x)
        q_copy = self.head_copy(x)

        q_pi = jax.lax.stop_gradient(q_old) + q_new - jax.lax.stop_gradient(q_copy)
        pred = jax.lax.select(jnp.asarray(inject, dtype=bool), q_pi, q_old)

        return QNetOutputs(q_1=pred, obs_rep=rep)
