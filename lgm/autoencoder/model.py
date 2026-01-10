from collections.abc import Callable

from torch import nn

from .likelihoods import loss_likelihood, map_likelihood
from ..types import DataBatchFloat, ScalarFloat


class Autoencoder(nn.Sequential):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 likelihood: str):
        """It's an autoencoder!

        Parameters:
            encoder: It's the encoder!
            decoder: It's the decoder! Wow!
            likelihood: The likelihood we assume for the data, which leads to a loss function equivalent to maximum
                        likelihood. Any of the ones listed in likelihoods.loss_likelihood function are valid.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood

    def forward(self,
                inputs: DataBatchFloat,
                map_to_expected: bool = False) -> DataBatchFloat:
        reconstructions = self.decoder(self.encoder(inputs))
        if map_to_expected:
            reconstructions = self.map_likelihood(reconstructions)
        return reconstructions

    def map_likelihood(self, 
                       outputs: DataBatchFloat) -> DataBatchFloat:
        """Converts network outputs (distribution parameters) to images outputs (expected values).

        Parameters:
            outputs: The values to convert.
        """
        return map_likelihood(outputs, self.likelihood)
        
    def loss_likelihood(self,
                        **additional_args) -> Callable[[DataBatchFloat, DataBatchFloat], ScalarFloat]:
        """Pick the correct loss for a given likelihood.

        Parameters:
            additional_args: Optional arguments for the loss besides outputs and targets.
                            For example, these could be constants needed for numerical stability.
        """
        return loss_likelihood(self.likelihood, **additional_args)
