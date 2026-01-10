import torch

from .model import ScoreModel
from ..common import TrainerBase, batched_multiply
from ..types import DataBatchFloat, LabelBatchFloat, ScalarFloat


class ScoreTrainer(TrainerBase[ScoreModel]):
    def core_step(self,
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Training step for score-based models.
        
        Note, this implements the loss from the "improved" paper, but with some simplifications. Technically, there
        should be a loss_weight sigma**2, a target_score = -(noisy_batch - input_batch) / sigma**2, and we would divide
        the model output by sigma (to implement conditioning on sigma). But you can pull the loss_weight into the
        difference, and then some terms cancel out. Also, the difference between noisy and input batch is just 
        sigma * epsilon. So we can cancel more stuff out there. We also ignore the constant 0.5 multiplier on the Fisher
        divergence, and average over data dimensions instead of summing (constant factor).
        """
        input_batch, _ = data_batch
        input_batch = input_batch.to(self.device)
        sampled_noise_index = torch.randint(high=self.model.num_sigmas, size=(input_batch.shape[0],))
        sigma = self.model.sigma_sequence[sampled_noise_index]
        epsilon = torch.randn_like(input_batch)
        noisy_batch = input_batch + batched_multiply(epsilon, sigma)

        #raise NotImplementedError("We need the score prediction from the 'noise conditional' score model, taking the "
        #                          "noisy samples as input.")
        score_prediction = self.model.noise_conditional_score(noisy_batch, sigma)
        #raise NotImplementedError("We need the target score, keeping in mind that the noisy samples follow a Gaussian "
        #                          "distribution centered around the data batch. Section 4.2 in the original paper should "
        #                          "be interesting for this.")
        target = -batched_multiply(epsilon, 1.0 / sigma)
        #raise NotImplementedError("We need the loss weights, see also section 4.2 in the first paper. NOTE that you can "
        #                          "also refer to equation (2) in the improved paper (top of page 3). This pulls the "
        #                          "weights into the squared difference and arrives at a slightly different formula, but "
        #                         "one that is mathematically equivalent. If you carefully consider what e.g. the "
        #                          "noise-conditional score model does (i.e. how it implements the conditioning), you "
        #                          "may be able to cancel out even more terms in that equation.")
        loss_weights = sigma**2
        loss = batched_multiply((score_prediction - target)**2, loss_weights).mean()
        return loss, {"fisher_divergence": loss}


    def plot_examples(self,
                      epoch_ind: int | None = None):
        with torch.inference_mode():
            generated = self.model.generate(self.plot_n_rows**2)
        self.plot_generated_grid(generated, epoch_ind)
