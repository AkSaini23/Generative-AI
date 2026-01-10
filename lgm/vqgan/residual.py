import torch
from torch import nn

from .model import VectorQuantizer
from ..types import DataBatchFloat, DataBatchFloatChannelsLast, IndexDataBatch


class ResidualVQ(nn.Module):
    def __init__(self,
                 n_codebooks: int,
                 **kwargs):
        """Residual vector quantizer for RVQVAE.

        Parameters:
            n_codebooks: How many codebooks/quantizers to create.
            kwargs: Arguments to VectorQuantizer.
        """
        super().__init__()
        self.n_codebooks = n_codebooks
        self.quantizers = nn.ModuleList([VectorQuantizer(**kwargs) for _ in range(n_codebooks)])

    def forward(self,
                inputs: DataBatchFloat,
                return_indices: bool = False) -> DataBatchFloat | tuple[DataBatchFloat, IndexDataBatch]:
        """Residual quantization with straight-through estimator.
        
        Parameters:
            inputs: As usual.
            return_indices: If True, return not just the quantized vectors, but also an "image" with the corresponding
                            codebook indices.
        """
        residual = inputs
        quantized_total = torch.zeros_like(inputs)
        if return_indices:
            all_indices = []
        for quantizer in self.quantizers:
            quantized_here = quantizer(residual, return_indices=return_indices)
            if return_indices:
                quantized_here, indices_here = quantized_here
                all_indices.append(indices_here)
            residual = residual - quantized_here
            quantized_total = quantized_total + quantized_here

        ste = inputs + (quantized_total - inputs).detach()
        if return_indices:
            closest_indices_image = torch.stack(all_indices, dim=1)
            return ste, closest_indices_image
        else:
            return ste

    def init_k_means(self,
                     inputs: DataBatchFloat):
        for quantizer in self.quantizers:
            quantizer.init_k_means(inputs)
            inputs = inputs - quantizer(inputs)
            
    def codebook_lookup(self,
                        indices: IndexDataBatch) -> DataBatchFloatChannelsLast:
        # TODO TYPING
        # accept channel-stack of indices, index each codebook and add
        """Turn codebook indices into the corresponding vectors."""
        embeddings = [self.quantizers[ind].codebook_lookup(indices[:, ind]) for ind in len(self.quantizers)]
        return torch.stack(embeddings).sum(axis=0)
