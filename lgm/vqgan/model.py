import torch
from scipy.cluster.vq import kmeans2
from torch import nn

from ..autoencoder import Autoencoder
from ..common import channels_first_to_last, channels_last_to_first, squared_distances
from ..gan import Discriminator, GAN
from ..types import DataBatchFloat, DataBatchFloatChannelsLast, IndexDataBatch, TabularBatchFloat, VectorBatchFloat


class VectorQuantizer(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 codebook_size: int,
                 gamma: float,
                 reset_frequency: int,
                 reset_threshold: float,
                 reset_inform: bool = False):
        """Standard vector quantizer for VQVAE.

        The codebook is updated using exponential moving average rather than a codebook loss.
        Also supports resetting unused vectors.
        NOTE things like updating or resetting the codebook is done inside the forward function if training is true for
        the module. This saves us from having to overwrite the training step. **However**, this is actually quite
        dangerous. If we ever run the model in training mode accidentally, it will incorrectly update the codebook and
        possibly even trigger resets. Thus, be extra careful to run the model in eval() mode when not training. Note
        that nn.Module is generally created in training mode, so best to call eval() directly.
        The funny thing here is that due to the combination with GAN training, we have to overwrite train_step anyway.
        So we take on a whole bunch of extra risk for literally no reward. But it's cool!
        
        Parameters:
            embedding_dim: Dimensionality of codebook vectors/encoder outputs.
            codebook_size: How many codebook vectors to create.
            gamma: Governs update speed of codebook EMA.
            reset_frequency: Codebook usage is checked, and vectors potentially reset, once per this many batches.
            reset_threshold: Vectors are reset when their usage falls below this threshold. Note that this should be
                             based on various parameters: Batch size, encoded image size, and codebook size. There are
                             batch_size * encoded_image_size vectors per batch; the codebook will distribute among
                             these. Thus we would expect each code to be used (batch_size * image_size / codebook_size)
                             many times. You will want to compute this quantity outside the class and pass some fraction
                             (say, 1/100 or so) of it to this function.
            reset_inform: If True, print information when a codebook reset is triggered.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.gamma = gamma
        self.register_buffer("codebook", torch.randn(codebook_size, embedding_dim))
        self.register_buffer("vector_averages", torch.zeros_like(self.codebook))
        self.register_buffer("n_averages", torch.zeros(codebook_size))
        self.reset_count = 0
        self.reset_frequency = reset_frequency
        self.reset_threshold = reset_threshold
        self.reset_inform = reset_inform

    def forward(self,
                inputs: DataBatchFloat,
                return_indices: bool = False) -> DataBatchFloat | tuple[DataBatchFloat, IndexDataBatch]:
        """Quantization with straight-through estimator.
        
        Parameters:
            inputs: As usual.
            return_indices: If True, return not just the quantized vectors, but also an "image" with the corresponding
                            codebook indices.
        """
        inputs_flat = channels_first_to_last(inputs).reshape(-1, self.embedding_dim)
        distances = squared_distances(inputs_flat, self.codebook)
        closest_indices = torch.argmin(distances, dim=1)
        closest_indices_image = closest_indices.view(-1, *inputs.shape[2:])
        quantized = self.codebook_lookup(closest_indices_image)
        quantized = channels_last_to_first(quantized)
        if self.training:
            self.update(inputs_flat, closest_indices)

        ste = inputs + (quantized - inputs).detach()
        if return_indices:
            return ste, closest_indices_image
        else:
            return ste
        
    def update(self,
               flat_vectors: TabularBatchFloat,
               indices: VectorBatchFloat):
        """Updates the codebook using exponential moving average."""
        with torch.no_grad():
            flat_indices_oh = torch.nn.functional.one_hot(indices, num_classes=self.codebook_size).to(torch.float32)
            update_vectors = flat_indices_oh.T @ flat_vectors
            update_usage = flat_indices_oh.sum(dim=0)
            self.vector_averages = self.gamma * self.vector_averages + (1 - self.gamma) * update_vectors
            self.n_averages = self.gamma * self.n_averages + (1 - self.gamma) * update_usage
            self.codebook = self.vector_averages / (self.n_averages[:, torch.newaxis] + 1e-8)
            self.reset_count += 1
            if self.reset_frequency is not None and not self.reset_count % self.reset_frequency:
                self.reset_unused(flat_vectors)

    def init_k_means(self,
                     inputs: DataBatchFloat):
        print("Initializing codebook with K-Means!")
        inputs_flat = channels_first_to_last(inputs).reshape(-1, self.embedding_dim)
        mean_vectors, assignments = kmeans2(inputs_flat.detach().cpu().numpy(), self.codebook_size, minit="points")
        device = self.codebook.device
        self.codebook = torch.tensor(mean_vectors, device=device)

        flat_indices_oh = torch.nn.functional.one_hot(torch.tensor(assignments, dtype=torch.int64,
                                                                   device=device).view(-1),
                                                      num_classes=self.codebook_size).to(torch.float32)
        update_vectors = flat_indices_oh.T @ inputs_flat
        update_usage = flat_indices_oh.sum(dim=0)
        self.vector_averages = update_vectors
        self.n_averages = update_usage
        self.reset_count = 0

    def reset_unused(self,
                     replacement_candidates: TabularBatchFloat):
        """Reset codebook vectors with too low usage to random reference vectors.
        
        Parameters:
            replacement_candidates: Batch of unquantized flattened encoder outputs. New values for codebook entries that
                                    should be reset are randomly picked from these.
        """
        reset_indices = torch.nonzero(self.n_averages < self.reset_threshold)[:, 0]
        if reset_indices.shape[0] > 0:
            replacement_indices = torch.randint(0, replacement_candidates.shape[0], size=(len(reset_indices),))
            replacements = replacement_candidates[replacement_indices]
            self.codebook[reset_indices] = replacements
            if self.reset_inform:
                print("triggered codebook reset for these indices:")
                print(reset_indices)
            self.vector_averages[reset_indices] = (self.codebook[reset_indices]
                                                   * (self.n_averages[reset_indices, torch.newaxis] + 1e-8))
            
    def codebook_lookup(self,
                        indices: IndexDataBatch) -> DataBatchFloatChannelsLast:
        """Turn codebook indices into the corresponding vectors."""
        return torch.nn.functional.embedding(indices, self.codebook)


class VQVAE(Autoencoder):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 quantizer: VectorQuantizer,
                 likelihood: str):
        """Vector-quantized autoencoder.

        Parameters:
            encoder: Typical encoder. Other than the standard VAE, this should *not* return twice as many outputs as
                     latent dimensions. No Gaussian sampling!
            decoder: Typical decoder.
            quantizer: Instance of VectorQuantizer.
            likelihood: As usual for autoencoders.
        """
        super().__init__(encoder, decoder, likelihood)
        self.quantizer = quantizer

    def forward(self,
                inputs: DataBatchFloat,
                map_to_expected: bool = True) -> DataBatchFloat:
        reconstructions = self.decoder(self.quantizer(self.encoder(inputs)))
        if map_to_expected:
            reconstructions = self.map_likelihood(reconstructions)
        return reconstructions
    
    def enquantize(self,
                   inputs: DataBatchFloat) -> IndexDataBatch:
        """Encode and quantize incoming data to codebook indices (not vectors)."""
        return self.quantizer(self.encoder(inputs), return_indices=True)[1]


class VQGAN(GAN):
    def __init__(self,
                 vqvae: VQVAE,
                 discriminator: Discriminator):
        """Container module for VQGAN training.
        
        Do not try to call this as a module.
        It's just a place to bundle vqvae and discriminator parameters.
        """
        # cursed... but I don't want to call GAN.__init__.
        # the inheritance from GAN is only due to the mess of generic types I have made.
        nn.Module.__init__(self)
        self.vqvae = vqvae
        self.discriminator = discriminator
