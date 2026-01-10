import torch

from .model import RBM
from ..common import TrainerBase
from ..types import DataBatchFloat, LabelBatchFloat, ScalarFloat, TabularBatchFloat

td = torch.distributions


class RBMTrainer(TrainerBase[RBM]):
    def __init__(self,
                 mode: str,
                 chain_length: int,
                 **kwargs):
        """Trainer for Binary (!) Restricted Boltzmann Machines.

        Parameters:
            mode: One of 'simple', 'cd', 'pcd': Which training algorithm to use. 'simple' starts new Markov chains to
                  sample from for each training step. 'cd' starts from the data batch. 'pcd' keeps a Markov chain
                  running throughout the entire training process.  cd and pcd should get away with fewer steps, i.e.
                  smaller chain_length. But in my experience, they don't end up producing good samples. :(
            chain_length: How long to run the Markov chains for training.
        """
        super().__init__(**kwargs)
        if mode not in ["simple", "cd", "pcd"]:
            raise ValueError
        self.mode = mode
        self.previous_v = None  # only used for pcd
        self.chain_length = chain_length
        
    def core_step(self,
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """RBM training core step.

        This handles generating hidden unit "data" as well as generating model samples, and then computing the 
        respective energies. Thus we cover both positive and negative phases.
        """
        input_batch, _ = data_batch
        v_data = input_batch.to(self.device).view(input_batch.shape[0], -1)
        with torch.no_grad():
            h_data = self.model.to_hidden_p(v_data)
            h_data = td.Bernoulli(probs=h_data).sample().to(v_data.device)
            
            # gibbs sampling -- naive/cd/pcd only differ in how the chains are initialized
            v_old = self.start_chain(v_data)
            v_new, h_new = self.model.gibbs_chain(v_old, self.chain_length)
            if self.mode == "pcd":
                self.previous_v = v_new
    
        # compute "loss" and take the gradient
        # although this loss can take any value (real number), it should converge to around 0
        # if it goes to -infinity there is something wrong!
        # most likely, your sampling is bad (e.g. not enough steps -> use longer chains)
        logits_positive = -self.model.energy(v_data, h_data)
        logits_negative = -self.model.energy(v_new, h_new)
        loss = -(logits_positive - logits_negative)
        return loss, {"rbm_loss": loss}

    def start_chain(self,
                    v_data: TabularBatchFloat) -> TabularBatchFloat:
        """Produce a starting point (visible data) for a Markov chain, depending on the training algorithm used.

        Parameters:
            v_data: Visible data used as a starting point for CD, or to get the shape for random samples.
        """
        if self.mode == "cd":
            return v_data
        elif self.mode == "pcd" and self.previous_v is not None:
            return self.previous_v
        else:  # simple OR first PCD step
            return self.model.start_sampler.sample((v_data.shape[0],)).to(v_data.device)

    def plot_examples(self,
                      epoch_ind: int | None = None):
        with torch.inference_mode():
            generated = self.model.generate(self.plot_n_rows**2, self.chain_length)
        self.plot_generated_grid(generated, epoch_ind)
