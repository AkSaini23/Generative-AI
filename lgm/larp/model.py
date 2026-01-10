from __future__ import annotations

import torch
from torch import nn

from ..common import channels_last_to_first
from ..types import DataBatchFloat, IndexSequenceBatch, SequenceBatchFloatChannelsLast
from ..vqgan import VQVAE


class LARP(nn.Module):
    def __init__(self,
                 autoregressor: ARTransformer,
                 vqvae: VQVAE,
                 latent_image_shape: tuple[int, ...]):
        """Model that autoregressively generates "images" of code indices, which are then passed to a VQVAE.
        
        Parameters:
            autoregressor: In principle, any autoregressive model that has a forward() for training and can generate.
                           We assume a Transformer because that's the only thing implemented in this repository. It's
                           also the only thing anyone uses anymore. :)
            vqvae: Vector-quantized autoencoder instance. Note that this should already be pretrained. The LARPTrainer
                   only trains the autoregressor.
            latent_image_shape: Since Transformers (or RNNs, for that matter) output flat sequences, we have to reshape
                                them to the correct size before putting them into the decoder.
        """
        super().__init__()
        self.autoregressor = autoregressor
        self.vqvae = vqvae
        self.latent_image_shape = latent_image_shape

    def generate(self,
                 n_samples: int,
                 prompt: IndexSequenceBatch | None = None,
                 temperature: float = 1.,
                 top_k: int | None = None) -> DataBatchFloat:
        """Generates codebook indices autoregressively and then decodes to data space.
        
        Parameters: As in ARTransformer.generate.
        """
        code_indices = self.autoregressor.generate(n_samples, prompt=prompt, temperature=temperature, top_k=top_k)
        code_indices = code_indices.view(-1, *self.latent_image_shape)
        codes = self.vqvae.quantizer.codebook_lookup(code_indices)
        codes = channels_last_to_first(codes)
        return self.vqvae.decoder(codes)


class ARTransformer(nn.Module):
    def __init__(self,
                 n_outputs: int,
                 d_model: int,
                 d_feedforward: int,
                 n_heads: int,
                 n_layers: int,
                 sequence_length: int):
        """Run-of-the-mill Transformer model.

        Note that this works on sequences, so if you want to generate images, you have to make sure they have been
        flattened beforehand. Also, we need a beginning of sequence token to start generation, so the input embedding
        gets one extra row. But since we never want to *predict* this special token, we do not have this in the output
        layer.
        
        Parameters:
            n_outputs: Number of outputs/classes. For a VQVAE latent model, should be equal to the codebook size.
            d_model: Transformer embedding & layer dimensionality.
            d_feedfoward: Dimensionality for the "mini MLPs" in each layer.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer layers.
            sequence_length: Length of the (flattened) sequences to generate. E.g. to generate 8x8 latent images, this
                             should be 64. Or 256 for 16x16, etc.
        """
        super().__init__()
        self.embedding = nn.Embedding(n_outputs + 1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_wavelen=sequence_length)
        transformer_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_feedforward, activation=nn.functional.gelu,
                                                       batch_first=True)
        self.body = nn.TransformerEncoder(transformer_layer, n_layers)
        self.output_layer = nn.Linear(d_model, n_outputs)
        self.body.apply(init_transformer)
        self.sequence_length = sequence_length

    def forward(self,
                inputs: IndexSequenceBatch,
                token_dropout: float = 0.) -> SequenceBatchFloatChannelsLast:
        mask = nn.Transformer.generate_square_subsequent_mask(inputs.shape[1], device=inputs.device)
        if token_dropout > 0. and self.training:
            random_values = torch.rand_like(mask)
            mask[random_values < token_dropout] = -torch.inf
        embedded = self.embedding(inputs + 1)
        pos_enc = self.positional_encoding(embedded)
        state = self.body(pos_enc, mask=mask)
        return self.output_layer(state)
    
    def generate(self,
                 n_samples: int,
                 prompt: IndexSequenceBatch | None = None,
                 temperature: float = 1.,
                 top_k: int | None = None) -> IndexSequenceBatch:
        """Good old sequential autoregressive generation.
        
        Parameters:
            n_samples: Number of samples to generate.
            prompt: If given, this is used as a start to the generation. Only the remaining time steps are generated
                    (e.g. if this has 20 time steps and self.sequence_length is 64, 44 will be generated) and the
                    full sequence is returned. Can be used to generate missing parts of data, e.g. generate the bottom
                    half on an image given the upper half. Note that we can only complete sequences following the
                    autoregressive ordering. E.g. if the model was trained on images top-left to bottom-right, you
                    cannot generate the top half given the bottom, only the other way around. Note that n_samples will
                    be ignored if this is provided.
            temperature: Logits are divided by this number. Pass a number < 1 to make the resulting categorical
                         distribution more peaked, which in turn should result in higher-quality generations, at the
                         cost of diversity. An extreme case is a temperature of 0, which corresponds to argmax sampling.
                         This is fully deterministic and not recommended for serious generations.
            top_k: If given, only the indices with the k largest outputs are considered for sampling. This removes any
                   chance of lower-probability vectors being sampled. Can be combined with temperature to further bias
                   towards the highest-probability vectors.
        """
        device = next(self.parameters()).device
        if prompt is not None:
            n_samples = prompt.shape[0]
            steps_to_generate = self.sequence_length - prompt.shape[1]
        else:
            steps_to_generate = self.sequence_length
        # -1 because we add 1 in the forward. so 0 is reserved for BOS character.
        previous_token = -1*torch.ones(n_samples, 1, dtype=torch.int64, device=device)
        if prompt is not None:
            previous_token = torch.cat((previous_token, prompt), axis=1)
        
        for _ in range(steps_to_generate):
            # [-1:] indexing preserves the time axis, so we don't have to re-add it later
            next_logits = self(previous_token)[:, -1:]
            if temperature == 0:
                random_draw = torch.argmax(next_logits, dim=-1)
            else:
                next_logits /= temperature
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_logits, k=top_k, sorted=False)
                    top_k_choice = torch.distributions.Categorical(logits=top_k_logits).sample()
                    # advanced technique: trial-and-error indexing :)
                    random_draw = torch.gather(top_k_indices[:, 0], 1, top_k_choice)
                else:
                    random_draw = torch.distributions.Categorical(logits=next_logits).sample()
            previous_token = torch.cat((previous_token, random_draw), axis=1)
        return previous_token[:, 1:]


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 1000,
                 max_wavelen=10000.0):
        """Classic Transformer sinusoidal positional encodings."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(max_wavelen)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0 ,:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self,
                x: SequenceBatchFloatChannelsLast) -> SequenceBatchFloatChannelsLast:
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)
    

def init_transformer(module: nn.Module):
    """Re-initialize Transfomer layers with random weights.
    
    The torch Transformer actually creates just one Transformer layer and then *copies* this multiple times, including
    weights. This means all layers start with the same initialization, which is a little bit weird. Thus, we manually
    set separate random weights for each layer.
    """
    if isinstance(module, nn.TransformerEncoderLayer):
        torch.nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
        torch.nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
        torch.nn.init.xavier_uniform_(module.linear1.weight)
        torch.nn.init.xavier_uniform_(module.linear2.weight)
        torch.nn.init.zeros_(module.linear1.bias)
        torch.nn.init.zeros_(module.linear2.bias)
