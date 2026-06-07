from flax import linen as nn

from .encoder import MinatarEncoder
from .helpers import normalize
from .output_types import FeatureNetOutputs


class MinatarFeatureNet(nn.Module):
    """
    A simple feature network for Minatar environments that encodes the input.
    """

    sf_dim: int

    def setup(self) -> None:
        self.encoder = MinatarEncoder()
        self.rep_hidden = nn.Dense(features=self.sf_dim)

    def __call__(self, obs):
        rep = self.encoder(obs)
        rep = rep.reshape((rep.shape[0], -1))

        rep_hidden = self.rep_hidden(rep)
        rep_hidden = nn.relu(rep_hidden)

        basis_features =  normalize()(rep_hidden)

        return FeatureNetOutputs(
            features=rep_hidden,
            basis_features=basis_features,
        )
