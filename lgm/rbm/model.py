import numpy as np
import torch
from torch import nn

from ..types import DataBatchFloat, TabularBatchFloat


class RBM(nn.Module):
    def __init__(self,
                 original_image_shape: tuple[int, ...],
                 n_hidden: int,
                 start_sampler: torch.distributions.Distribution | None = None):
        """Binary RBM class.
        
        Parameters:
            original_image_shape: RBMs work on vector data. This means images are flattened. This should give the actual
                                  shape of the images, so we can reshape our flat generations to the appropriate shape.
            n_hidden: Number of hidden units to use.
            start_sampler: A distribution we can sample from to start the Markov chains.
        """
        super().__init__()
        self.original_image_shape = original_image_shape
        n_visible = np.prod(original_image_shape)
        self.w_v_to_h = nn.Parameter(torch.zeros(n_visible, n_hidden))
        self.bias_v = nn.Parameter(torch.zeros(n_visible))
        self.bias_h = nn.Parameter(torch.zeros(n_hidden))
        if start_sampler is None:
            start_sampler = torch.distributions.Bernoulli(probs=torch.full((n_visible,), fill_value=0.5))
        self.start_sampler = start_sampler
    
    def to_hidden_p(self,
                    visible: TabularBatchFloat) -> TabularBatchFloat:
        """Get conditional probabilities p(h|v)."""
        return nn.functional.sigmoid(visible @ self.w_v_to_h + self.bias_h)

    def to_visible_p(self,
                     hidden: TabularBatchFloat) -> TabularBatchFloat:
        """Get conditional probabilities p(v|h)."""
        return nn.functional.sigmoid(hidden @ self.w_v_to_h.T + self.bias_v)

    def generate(self,
                 n_generations: int,
                 chain_length: int,
                 return_probs: bool = True,
                 return_chain: bool = False) -> DataBatchFloat | tuple[DataBatchFloat, list[TabularBatchFloat]]:
        """Create samples from the RBM distribution via Markov chains.
        
        Parameters:
            n_generations: How many independent chains to run (i.e. how many images do you want).
            chain_length: Markov Chain length.
            return_probs: If True, we return the visible probabilities for the last p(v|h) instead of sampling. This
                          gives much smoother, less noisy samples.
            return_chain: If True, we also return the entire chain of visible samples as a list. Note that even if 
                          return_probs is True, this will contain the *binary samples* for the last step.
        """
        initial_v = self.start_sampler.sample((n_generations,)).to(self.w_v_to_h.device)
        chain_outputs = self.gibbs_chain(initial_v, chain_length, return_v_probs=return_probs,
                                         return_chain=return_chain)
        
        final_v = chain_outputs[0].view(-1, *self.original_image_shape)
        if return_chain:
            return final_v, chain_outputs[2]
        else:
            return final_v

    def gibbs_chain(self,
                    initial_visible: TabularBatchFloat,
                    n_steps: int,
                    return_v_probs: bool = False,
                    return_chain: bool = False) -> tuple[TabularBatchFloat, TabularBatchFloat] | tuple[TabularBatchFloat, TabularBatchFloat, list[TabularBatchFloat]]:
        """Run Markov Chains using Gibbs Sampling.
        
        Parameters:
            initial_visible: b x d batch of chain starting points. 
                             We run b chains in parallel, returning one output per chain.
            n_steps: How many Gibbs Sampling steps to run. 
                     Note that one step means sampling from p(h|v) *and* then p(v|h).
            return_v_probs, return_chain: See function generate().
        """
        new_v = initial_visible.clone()
        if return_chain:
            full_chain = [new_v]
        for _ in range(n_steps):
            new_v, new_h = self.gibbs_update(new_v)
            if return_chain:
                full_chain.append(new_v)

        if return_v_probs:
            new_v = self.to_visible_p(new_h)
        if return_chain:
            return new_v, new_h, full_chain
        else:
            return new_v, new_h

    def gibbs_update(self,
                     visible: TabularBatchFloat) -> tuple[TabularBatchFloat, TabularBatchFloat]:
        """A single Gibbs update step, returning both new h and v values."""
        p_h = self.to_hidden_p(visible)
        new_h = torch.distributions.Bernoulli(probs=p_h).sample().to(visible.device)
        p_v = self.to_visible_p(new_h)
        new_v = torch.distributions.Bernoulli(probs=p_v).sample().to(visible.device)
        return new_v, new_h

    def energy(self,
               visible: TabularBatchFloat,
               hidden: TabularBatchFloat) -> TabularBatchFloat:
        """Our friendly RBM Energy function, implemented for *batches* of v and h."""
        return -(visible @ self.bias_v
                 + hidden @ self.bias_h
                 + ((visible @ self.w_v_to_h) * hidden).sum(dim=1)).mean()
