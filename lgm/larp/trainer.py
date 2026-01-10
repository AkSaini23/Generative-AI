import torch
from torch import nn

from .model import LARP
from ..common import TrainerBase, channels_last_to_first
from ..evaluation import accuracy_sequence, top_k_accuracy_sequence
from ..types import DataBatchFloat, LabelBatchFloat, ScalarFloat


class LARPTrainer(TrainerBase[LARP]):
    def __init__(self,
                 encode_on_the_fly: bool = True,
                 label_smoothing: float = 0.,
                 token_dropout: float = 0.,
                 accuracy_k: int = 5,
                 **kwargs):
        """Trainer for LARP model.
        
        Assumes pretrained VQVAE. This only trains the autoregressive part.

        Parameters:
            encode_on_the_fly: If True, we assume the data is "raw" and must be encoded by the VQVAE. This is then done
                               at each training step. Otherwise, we assume the data has already been encoded and
                               quantized, and the data consists of the codebook indices. While this option is certainly
                               convenient, it also significantly slows down training. It is recommended that you
                               manually encode the training and validation data beforehand, so the VQVAE only needs to
                               be run once per data point.
            label_smoothing: Passed to cross-entropy loss.
            token_dropout: During training, we may randomly drop out parts of the self-attention matrix, forcing the
                           model to rely on other tokens. This should reduce overfitting in the attention patterns.
            accuracy_k: Value k used in top-k accuracy.
        """
        super().__init__(**kwargs)
        self.encode_on_the_fly = encode_on_the_fly
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.token_dropout = token_dropout
        self.accuracy_k = accuracy_k

    def core_step(self, 
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Standard AR training step.
        
        Data is optionally first encoded to the quantized latent space. Note that we add a "beginning of sequence"
        token to each input sequence, as we need something to start the generation process with. We encode this by
        -1, and then inside the transformer call, 1 is added to all inputs -- so it will end up at index 0, while the
        original codebook vector with index 0 becomes 1.
        """
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        if self.encode_on_the_fly:
            self.model.vqvae.eval()
            with torch.no_grad():
                input_batch = self.model.vqvae.enquantize(input_batch)
        
        input_batch = input_batch.view(input_batch.shape[0], -1)
        with_start_token = torch.cat((-1*torch.ones(input_batch.shape[0], 1, dtype=torch.int64, device=self.device),
                                      input_batch[:, :-1]), dim=1)
        predictions = self.model.autoregressor(with_start_token, self.token_dropout)
        predictions = channels_last_to_first(predictions)
        cross_entropy = self.loss_fn(predictions, input_batch)
        with torch.inference_mode():
            batch_accuracy = accuracy_sequence(predictions, input_batch)
            batch_top_k_accuracy = top_k_accuracy_sequence(predictions, input_batch, k=5)
        return cross_entropy, {"cross_entropy": cross_entropy, "accuracy": batch_accuracy,
                               f"top_{self.accuracy_k}_accuracy": batch_top_k_accuracy}
    
    def plot_examples(self,
                      epoch_ind: int | None = None):
        with torch.inference_mode():
            generated = self.model.generate(self.plot_n_rows**2)
        self.plot_generated_grid(generated, epoch_ind)
