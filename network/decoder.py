from flax import linen as nn
import jax.numpy as jnp


class Decoder(nn.Module):
    """
    A simple decoder for reconstructing states or images from a latent representation.
    """

    sf_dim: int
    num_channels: int = 32
    use_framestack: bool = False
    num_stacked_frames: int = 4

    @nn.compact
    def __call__(
        self, representation: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
        actions_embedding = nn.Dense(features=self.sf_dim)(actions)
        rep_embedding = nn.Dense(features=self.sf_dim)(representation)

        rep_action_embedding = jnp.multiply(rep_embedding, actions_embedding)
        rep_action_embedding = nn.Dense(features=self.sf_dim)(rep_action_embedding)

        # 41472 is derived from the output of the encoder after conv
        rep_action_embedding = nn.Dense(features=41472)(rep_action_embedding)

        rep_action_embedding = nn.relu(rep_action_embedding)
        rep_action = jnp.reshape(rep_action_embedding, (-1, 36, 36, self.num_channels))

        deconv_output_1 = nn.ConvTranspose(
            features=self.num_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
        )(rep_action)
        deconv_output_1 = nn.relu(deconv_output_1)

        deconv_output_2 = nn.ConvTranspose(
            features=self.num_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
        )(deconv_output_1)
        deconv_output_2 = nn.relu(deconv_output_2)

        deconv_output_3 = nn.ConvTranspose(
            features=self.num_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
        )(deconv_output_2)
        deconv_output_3 = nn.relu(deconv_output_3)

        if self.use_framestack:
            deconv_output_4 = nn.ConvTranspose(
                features=self.num_stacked_frames,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
            )(deconv_output_3)

        else:
            deconv_output_4 = nn.ConvTranspose(
                features=3,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
            )(deconv_output_3)

        deconv_output_4 = nn.sigmoid(
            deconv_output_4
        )  # output is between 0 and 1 for pixels
        return deconv_output_4
