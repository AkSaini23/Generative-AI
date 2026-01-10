from __future__ import annotations

import torch
from torch import nn

from .legacy import GaussSampler
from ..autoencoder import Autoencoder
from ..types import AnyBatchFloat, DataBatchFloat, TabularBatchFloat


class ConditionalVAEGauss(Autoencoder):
    def __init__(self,
                 encoder: ConditionalNet,
                 decoder: ConditionalNet,
                 conditioning_net: nn.Module | None,
                 likelihood: str,
                 prior_dim: tuple[int, ...],
                 n_classes: int | None):
        """Wrapper for VAE with Gaussian prior/posterior.

        If conditioning_net is given, this will be a conditional model!

        Parameters:
            encoder: Takes inputs and returns *parameters* for latent distributions, i.e. q(z|x). Note, since we are
                     using a Gaussian q, the encoder should output *twice as many values* as our desired latent space
                     dimensionality.
            decoder: Takes code samples and reconstructs the inputs. Make sure this returns the correct shape. Final
                     layer should have no activation function, as the values are mapped to the correct range using the
                     map_likelihood function.
            conditoning_net: Model that takes conditoning information (such as class labels) and returns a deep
                             representation ("embedding"). A typical structure would be a OneHot layer followed by some
                             Linear layers, or simply an Embedding layer. If None, this will be an unconditional model.
            likelihood: As given to the Trainer object -- which likelihood we use determines how the decoder outputs
                        are interpreted.
            prior_dim: Since torch modules are not aware of their own input size, we need to store this somewhere so we
                       can generate new samples. Should be the shape of the input that the decoder expects.
            n_classes: For similar reasons as prior_dim, we need to tell the model how many classes it "knows". Because
                       it doesn't actually know otherwise.
        """
        super().__init__(encoder, decoder, likelihood)
        self.encoder = encoder
        self.decoder = decoder
        self.conditioning_net = conditioning_net
        self.sampler = GaussSampler()
        self.likelihood = likelihood
        self.prior_dim = prior_dim
        
        self.is_conditional = conditioning_net is not None
        self.n_classes = n_classes

    def forward(self, 
                inputs: DataBatchFloat,
                conditioning: AnyBatchFloat | None,
                map_to_expected: bool = True) -> DataBatchFloat:
        embedded_conditioning = self.conditioning_net(conditioning) if self.is_conditional else None
        reconstructions = self.decoder(self.sampler(*self.encoding_parameters(inputs, embedded_conditioning)),
                                       embedded_conditioning)
        if map_to_expected:
            reconstructions = self.map_likelihood(reconstructions)
        return reconstructions

    def encoding_parameters(self,
                            inputs: DataBatchFloat,
                            embedded_conditioning: TabularBatchFloat | None) -> tuple[AnyBatchFloat, AnyBatchFloat]:
        """Split encoder outputs into separate means and log variances.
        
        IMPORTANT: The conditioning argument needs to already have the conditioning net applied. This is due to the
        context where this function is applied (in training): We have to pass the conditioning into multiple functions,
        so we try to reduce computational waste by re-using the conditioning net outputs.
        """
        parameters = self.encoder(inputs, embedded_conditioning)
        split_size = parameters.shape[1] // 2
        mean, log_variance = torch.split(parameters, split_size, dim=1)
        return mean, log_variance

    def generate(self, 
                 n_samples: int | None = None,
                 conditioning:AnyBatchFloat | None = None) -> DataBatchFloat:
        """Do conditional or unconditional generation, depending on the model.
        
        Parameters:
            n_samples: How many samples to generate. Only used if this is an unconditional model, else it will be
                       ignored and you MUST supply the conditioning argument instead.
            conditioning: Conditioning input to generate with. Will create one sample per element (row) in this
                          argument. MUST be supplied for conditional models, ignored for unconditional ones. This must
                          be "raw", i.e. NOT already processed by the conditioning net.
        """
        device = next(self.parameters()).device
        if self.is_conditional and conditioning is None:
            raise ValueError("Trying to generate unconditionally for a conditional model.")
        elif self.is_conditional and n_samples is not None:
            print("Warning! n_samples supplied to conditional model; argument will be ignored!")
        elif not self.is_conditional and conditioning is not None:
            print("Warning! conditioning supplied to unconditional model; argument will be ignored!")

        if conditioning is not None:
            conditioning = conditioning.to(device)
            embedded_conditioning = self.conditioning_net(conditioning)
            codes = torch.randn(conditioning.shape[0], *self.prior_dim, device=device)
        else:
            embedded_conditioning = None
            codes = torch.randn(n_samples, *self.prior_dim, device=device)
        return self.map_likelihood(self.decoder(codes, embedded_conditioning))


class ConditionalNet(nn.Module):
    def __init__(self,
                 root: nn.Module,
                 body: nn.Module,
                 head: nn.Module):
        """General network structure with conditioning input.
        
        We cannot use Sequential due to the multiple inputs.
        NOTE!! This is expected to be used as component of a larger module that computes correct conditioning (e.g. via
        an embedding). This module will directly supply the conditioning to the body as-is. You can also use this for
        unconditional models! In that case you must explicitly pass None to the conditioning in the forward call (no
        default argument!).

        Parameters:
            root: Any layers that should be applied before the body, such as an initial convolution. These will not
                  receive conditioning.
            body: For conditional models: Should be a CNNBody able to take conditional inputs (cond_dim > 0).
                  For unconditional models: Use standard CNNBody without conditional inputs.
            head: A module that is applied after the body. This will also not receive conditioning. For the encoder,
                  this maybe some Flatten + Linear layers. For the decoder, you likely want a convolutional layer
                  mapping back to the number of channels in the data.
        """
        super().__init__()
        self.root = root
        self.body = body
        self.head = head

    def forward(self,
                inputs: DataBatchFloat,
                embedded_conditioning: TabularBatchFloat | None) -> AnyBatchFloat:
        outputs = self.root(inputs)
        outputs = self.body(outputs, conditioning=embedded_conditioning)
        return self.head(outputs)
