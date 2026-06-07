import flax.linen as nn
import jax.numpy as jnp
from .encoder import MinatarEncoder
from .output_types import QNetBeakersOutputs
import jax
import chex


class QConsolidationNetworkMinatar(nn.Module):
    """
    A simple Q-network for Minatar environments.
    """

    feature_dim: int
    num_actions: int
    num_beakers: int

    def setup(self) -> None:
        self.encoder = MinatarEncoder()

        self.q_consolidation_1 = {
            f"q_consolidation1_{i}": nn.Dense(self.feature_dim)
            for i in range(self.num_beakers)
        }

        self.q_consolidation_2 = {
            f"q_consolidation2_{i}": nn.Dense(self.num_actions)
            for i in range(self.num_beakers)
        }

    @nn.compact
    def make_q_reps(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=self.feature_dim)(x)
        x = nn.relu(x)
        return x

    def q_transform(self, features, idx):
        key_q_consolidation1 = f"q_consolidation1_{idx}"
        key_q_consolidation2 = f"q_consolidation2_{idx}"

        q = self.q_consolidation_1[key_q_consolidation1](features)
        q = nn.relu(q)
        q = self.q_consolidation_2[key_q_consolidation2](q)
        return jnp.reshape(q, (-1, self.num_actions))

    def __call__(self, x):
        batch_size = x.shape[0]
        rep = self.encoder(x)
        features = rep.reshape((rep.shape[0], -1))
        features_no_grad = jax.lax.stop_gradient(features)

        q_current_beaker_0 = self.q_transform(features, 0)

        """
        Here, we use the set of q where all beakers have grad and uses the features_no_grad. In addition, 
        the first beaker also uses features_no_grad so that the encoder is not updated. This is for the consolidation loss.  
        """
        q_current_beaker_0_stop_grad_feature = self.q_transform(features_no_grad, 0)
        q_list_consolidation = [
            self.q_transform(features_no_grad, i) for i in range(1, self.num_beakers)
        ]

        q_stacked_with_grad_for_all_q = jnp.stack(
            q_list_consolidation, axis=1
        )  # (batch_size, num_beakers, num_actions)

        chex.assert_shape(
            q_stacked_with_grad_for_all_q,
            (
                batch_size,
                self.num_beakers - 1,  # exclude the first beaker
                self.num_actions,
            ),
        )

        return QNetBeakersOutputs(
            q_current_beaker_0_stop_grad_feature = jnp.expand_dims(q_current_beaker_0_stop_grad_feature, axis=1), # for consolidation loss
            q_1=q_current_beaker_0,  # (batch_size, num_actions)
            q_vals_consolidation=q_stacked_with_grad_for_all_q,
        )
