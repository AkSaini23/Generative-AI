from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import chain
from time import perf_counter
from typing import Generic, TypeVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..types import DataBatchFloat, LabelBatchFloat, ScalarFloat
from ..visualization import plot_audio_grid, plot_image_grid


Model = TypeVar("Model", bound=nn.Module)


class TrainerBase(Generic[Model]):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 training_loader: DataLoader[tuple[DataBatchFloat, LabelBatchFloat]], 
                 validation_loader: DataLoader[tuple[DataBatchFloat, LabelBatchFloat]],
                 n_epochs: int,
                 device: str,
                 plot_every_n_epochs: int | None,
                 plot_figsize: tuple[int, int] = (12, 12),
                 plot_n_rows: int = 10,
                 plot_descale: Callable[[DataBatchFloat], DataBatchFloat] | None = None,
                 modality: str = "image",
                 sampling_rate: int | None = None,
                 decay_style: str = "epoch",
                 early_stopper: EarlyStopping | None = None, 
                 ema: EMA | None = None,
                 checkpointer: Checkpointer | None = None,
                 verbose: bool = True,
                 use_tqdm: bool = False,
                 tensorboard_logdir: str | None = None,
                 tensorboard_figures: bool = False,
                 suppress_plots: bool = False):
        """Base class for training generative models.

        Any Trainer for a specific kind of model should inherit fromt his and implement the core_step function.

        Parameters:
            model: The model to train.
            optimizer: Guess what.
            scheduler: Learning rate schedulers to apply.
            training_loader, validation_loader: Dataloaders for training/validation sets.
            n_epochs: Number of full iterations over the training loader.
            device: Device on which all the torch stuff should happen (e.g. "cuda").
            plot_every_n_epochs: Every so often, it makes sense to e.g. plot some generated images from the model. This
                                 allows us too judge training progress visually. The Trainer class should implement the
                                 plot_examples method. Pass None to disable plotting.
            plot_figsize: Figure size for regular plots.
            plot_n_rows: Usually, we will generate n x n generations each time we plot something.
            plot_descale: See same argument in lgm.data.get_datasets_and_loaders.
            modality: One of 'image', 'audio'. Depending on this, we choose how to present generated samples if plotting
                      is active.
            sampling_rate: If modality is 'audio', must be the correct sampling rate.
            decay_style: One 'epoch' or 'step'. The former applies the learning rate scheduler after each epoch. The
                         latter applies it after each training step. You are responsible for making sure the scheduler
                         uses a sensible number of steps for each! NOTE, a ReduceLROnPlateau scheduler will always
                         trigger after the epoch, since it needs evaluation results.
            early_stopper: Optional early stopping object. Pass None to disable early stopping.
            ema: If given, create a storage of exponential moving averages (EMA) of the model parameters after each
                 training step. These can later be used to overwrite the trained model parameters. Supposedly, this
                 helps sometimes. =) It's especially popular with score-based/diffusion models. Pass None to disable 
                 this. NOTE you will have to manually apply the EMA parameters after training.
            checkpointer: If given, checkpoints will be stored at the desired frequency (determined by the checkpoint
                          object). In addition, we will save a checkpoint with _final suffix at the end of training.
            verbose: If True, report on training progress throughout.
            use_tqdm: If True, and verbose is also True, supply per-epoch progress bars.
            tensorboard_logdir: If given, will log training/validation losses to the specified directory for
                                visualization with TensorBoard. Pass None to disable logging.
                                Note that you can supply additional logging in subclasses core_step or plot_examples
                                functions. For example, you could also log the images created in the latter.
            tensorboard_figures: If True, save figures generated in plot_examples to tensorboard logs. Does nothing if
                                 tensorboard_logdir is not given. Note that this is not magic, so any plot_examples
                                 implementations need to make sure this actually happens. This argument merely provides
                                 a switch to turn this on or off.
            suppress_plots: If True, and tensorboard_figures is True, figures will *only* be stored in tensorboard, and
                            not plotted to output (e.g. in a notebook). No effect if tensorboard_figures is False.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.n_epochs = n_epochs
        self.device = device

        self.plot_every_n_epochs = plot_every_n_epochs
        self.plot_figsize = plot_figsize
        self.plot_n_rows = plot_n_rows
        self.plot_descale = plot_descale

        self.modality = modality
        self.sampling_rate = sampling_rate
        if modality == "audio" and sampling_rate is None:
            raise ValueError("If modality is 'audio', sampling_rate must be given.")
        
        self.decay_style = decay_style
        if decay_style not in ["epoch", "step"]:
            raise ValueError(f"Invalid decay_style {decay_style}. Valid are 'epoch', 'step'.")
        self.early_stopper = early_stopper
        self.ema = ema
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.checkpointer = checkpointer

        if tensorboard_logdir is not None:
            self.writer = SummaryWriter(tensorboard_logdir)
        else:
            self.writer = None
        self.tensorboard_figures = tensorboard_figures
        self.suppress_plots = suppress_plots

    def train_model(self) -> defaultdict[str, np.ndarray]:
        """The main training & evaluation loop + housekeeping.

        Returns:
            Dictionary with training and evaluation metrics per epoch. This maps each train/val metric name to a numpy
            array of per-epoch results. NOTE, for training metrics we only track the average over the epoch.
            This is somewhat imprecise, as the model changes over the epoch. So the metrics at the end of the epoch will
            usually be better than at the start, but we average over everything. A "clean" approach would evaluate on
            the training set at the end of each epoch, but this would take significantly longer.
        """
        n_training_examples = len(self.training_loader.dataset)
        batches_per_epoch = n_training_examples // self.training_loader.batch_size
        print(f"Running {self.n_epochs} epochs at {batches_per_epoch} steps per epoch.")
        
        full_metrics = defaultdict(list)
        self.optimizer.zero_grad()  # safety first!
        for epoch_ind in tqdm(iterable=range(self.n_epochs), desc="Overall progress", leave=True,
                          disable=not self.use_tqdm or not self.verbose):
            if self.plot_every_n_epochs is not None and not epoch_ind % self.plot_every_n_epochs:
                self.model.eval()
                self.plot_examples(epoch_ind)
            epoch_train_metrics = self.train_epoch(epoch_ind)
            should_stop = self.finish_epoch(full_metrics, epoch_train_metrics, epoch_ind)
            if should_stop:
                print("Early stopping...")
                break

        if self.checkpointer is not None:
            final_path = os.path.join(self.checkpointer.directory, self.checkpointer.checkpoint_name + "_final.pt")
            torch.save(self.model.state_dict(), final_path)
        self.model.eval()
        # TODO make this prettier
        for key in full_metrics:
            full_metrics[key] = np.array(full_metrics[key])
        return full_metrics
    
    def train_epoch(self,
                    epoch_ind: int) -> defaultdict[str, list[float]]:
        """One epoch training loop. Iterates over the training dataloader once and collects metrics.

        Returns:
            Dictionary mapping metric names to lists of per-batch results.
        """
        if self.verbose:
            print(f"Starting epoch {epoch_ind + 1}...", end=" ")
        start_time = perf_counter()
        epoch_train_metrics = defaultdict(list)
        
        self.model.train()
        # manual progressbar required due to multiprocessing in dataloaders
        with tqdm(total=len(self.training_loader), desc=f"Training", leave=False,
                    disable=not self.use_tqdm or not self.verbose) as progressbar:
            for data_batch in self.training_loader:
                batch_losses = self.train_step(data_batch)
                for key in batch_losses:
                    epoch_train_metrics[key].append(batch_losses[key].item())
                if (self.decay_style == "step"
                    and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                    self.scheduler.step()
                progressbar.update(1)
        
        end_time = perf_counter()
        time_taken = end_time - start_time
        if self.verbose:
             print(f"\tTime taken: {time_taken:.4g} seconds")
        return epoch_train_metrics
    
    def finish_epoch(self,
                     full_run_metrics: dict[str, list[float]],
                     epoch_train_metrics: dict[str, list[float]],
                     epoch_ind: int) -> bool:
        """Bunch of housekeeping after each epoch training loop.
        
        This function:
            - Evaluates on the validation set.
            - Applies learning rate scheduling.
            - Checks for early stopping.
            - Collects train and validation metrics in one place.
            - Optionally writes Tensorboard summaries.

        Parameters:
            full_run_metrics: Should be the dictionary created at the start of train_model. This is modified in-place
                              inside this function.
                              TODO perhaps refactor this into an attribute.
            epoch_train_metrics: As returned from the last train_epoch call.
            epoch_ind: The index of the epoch (wow).

        Returns:
            Boolean flag from early stopping.
        """
        val_loss_full, val_losses = self.evaluate()
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss_full)
        elif self.decay_style == "epoch":
            self.scheduler.step()
                
        if self.early_stopper is not None:
            should_stop = self.early_stopper.update(val_loss_full)
        else:
            should_stop = False

        for key in epoch_train_metrics:
            val_metric = val_losses[key].item()
            train_metric = np.mean(epoch_train_metrics[key])

            full_run_metrics["val_" + key].append(val_metric)
            full_run_metrics["train_" + key].append(train_metric)
            if self.writer is not None:
                self.writer.add_scalars(key, {"training": train_metric, "validation": val_metric}, epoch_ind)

        if self.verbose:
            print("\tMetrics:")
            for key in full_run_metrics:
                print(f"\t\t{key}: {full_run_metrics[key][-1]:.6g}")
            print(f"\tLR is now {self.scheduler.get_last_lr()[0]:.10g}")
        print()
        if self.writer is not None:
            self.writer.flush()
        if self.checkpointer is not None:
            self.checkpointer.maybe_checkpoint(epoch_ind)
        return should_stop

    def evaluate(self) -> tuple[ScalarFloat, defaultdict[str, ScalarFloat]]:
        """One evaluation loop.

        Returns:
            - The full evaluation loss (e.g. for early stopping)
            - Dictionary with separate metrics. Note that these are tensors, not np arrays like in the train_model
              function.
        """
        self.model.eval()
        num_batches = len(self.validation_loader)
        val_loss_full = 0.
        val_losses = defaultdict(float)

        # manual progressbar required due to multiprocessing in dataloaders
        with tqdm(total=len(self.validation_loader), desc="Validation", leave=False,
                  disable=not self.use_tqdm or not self.verbose) as progressbar:
            for data_batch in self.validation_loader:
                batch_loss_full, batch_losses = self.eval_step(data_batch)
                
                val_loss_full += batch_loss_full
                for key in batch_losses:
                    val_losses[key] += batch_losses[key]
                progressbar.update(1)
                
        # TODO this is not correct as the last batch may be smaller -> slight bias
        val_loss_full /= num_batches
        for key in val_losses:
            val_losses[key] /= num_batches
        return val_loss_full, val_losses
    
    def train_step(self,
                   data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> dict[str, ScalarFloat]:
        """Standard training step: Get loss (through core_step), backpropagate, apply gradients.
        
        Parameters:
            data_batch: A tuple of images, labels.
        """
        batch_loss_full, batch_losses = self.core_step(data_batch)
        batch_loss_full.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.ema is not None:
            self.ema.update()
        return batch_losses
    
    def eval_step(self,
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Compute loss for validation. No gradients are computed!"""
        with torch.inference_mode():
            batch_loss_full, batch_losses = self.core_step(data_batch)
        return batch_loss_full, batch_losses

    def core_step(self,
                  data_batch: tuple[DataBatchFloat, LabelBatchFloat]) -> tuple[ScalarFloat, dict[str, ScalarFloat]]:
        """Main logic for computing losses. Not implemented as it is model-dependent.
        
        Generally this function should:
        - Split data into inputs/labels
        - Move data to the correct device
        - Apply the model
        - Compute any losses
        - Return the overall loss, and a dictionary with component losses (e.g. for VAE: reconstruction and KLD losses).

        Parameters:
            data_batch: Expected to be a tuple for inputs, labels. Generative modeling doesn't need labels, but we later
                        want to implement conditional models that do. So we just carry the labels around in the dataset.
                        For this reason some of the datasets we use may have "dummy" labels to avoid mismatches.

        Return:
            A tuple. The first entry should be the full loss to be used for training, or things like early stopping.
            Many models will only have one loss, but some have multiple. This will likely be their sum. The second entry
            should be a dictionary mapping names to the individual loss components. Even if you have only one loss in
            the model, just repeat it here.
        """
        raise NotImplementedError

    def plot_examples(self,
                      epoch_ind: int | None = None):
        """This function is called every couple epochs. You can really do whatever you want in here.
        
        But it is intended to visually show model progress, e.g. through plotting some generated samples.
        """
        pass

    def plot_generated_grid(self,
                            generated: DataBatchFloat,
                            epoch_ind: int | None = None,
                            title: str = "Generations",
                            subtitles: Iterable[str] | None = None):
        """Standard function to display generated data (images or audio).
        
        Although the base function plots nothing, this function is used over and over again in sublcasses. So it makes
        sense to have it in the base class to prevent copy-paste.

        Parameters:
            generated: Batch of data we want to display. We assume that this has self.plot_n_rows**2 many rows.
            epoch_ind: Only necessary for Tensorboard logging.
            title: Sets (part of, in some cases) the figure title.
        """
        # TODO not very elegant
        if self.modality == "image":
            plot_image_grid(generated,
                            figure_size=self.plot_figsize, n_rows=self.plot_n_rows, title=title, subtitles=subtitles,
                            plot_descale=self.plot_descale, writer=self.writer, epoch_ind=epoch_ind,
                            tensorboard_figures=self.tensorboard_figures, suppress_plots=self.suppress_plots)
        elif self.modality  == "audio":
            plot_audio_grid(generated,
                            figure_size=self.plot_figsize, n_rows=self.plot_n_rows, title=title, subtitles=subtitles,
                            plot_descale=self.plot_descale, sampling_rate=self.sampling_rate, writer=self.writer,
                            epoch_ind=epoch_ind, tensorboard_figures=self.tensorboard_figures,
                            suppress_plots=self.suppress_plots)
        else:
            raise NotImplementedError(f"Modality {self.modality} not implemented for plotting")


class ParameterTracker:
    def __init__(self,
                 model: nn.Module,
                 trainable_only: bool = False,
                 include_buffers: bool = True):
        """Base class for parameter trackers/storages like early stopping or Polyak averaging.

        Parameters:
            model: Model to track.
            trainable_only: If True, we only create EMAs for trainable parameters. Otherwise, anything in the model
                            state dict will be affected.
            include_buffers: If True, also create EMAs for buffers (like Batchnorm moving averages). Only float-type
                             buffers are used!
        """
        self.model = model
        self.trainable_only = trainable_only
        self.include_buffers = include_buffers
        self.tracked_parameters = [param.detach().clone() for param in self.get_parameters()]
        self.backup = None

    def get_parameters(self) -> Iterable[torch.Tensor]:
        """Return all desired parameters."""
        if not self.include_buffers:
            return iter(param for param in self.model.parameters() if param.requires_grad or not self.trainable_only)
        return iter(param for param in chain(self.model.parameters(), self.model.buffers())
                    if (param.requires_grad or not self.trainable_only) and torch.is_floating_point(param))
    
    @torch.no_grad()
    def apply_parameters(self):
        """Overwrite model parameters with EMA parameters while also creating a backup."""
        self.make_backup()
        for tracked_param, model_param in zip(self.tracked_parameters, self.get_parameters()):
            model_param[:] = tracked_param

    def make_backup(self):
        """Make a backup of original model parameters."""
        if self.backup is None:
            print("Creating backup...")
            self.backup = [param.detach().clone() for param in self.get_parameters()]
        else:
            print("backup has been created already! This backup has been SKIPPED.")

    @torch.no_grad()
    def apply_backup(self):
        """Restore backed up model parameters."""
        for backup_param, model_param in zip(self.backup, self.get_parameters()):
            model_param[:] = backup_param


class EarlyStopping(ParameterTracker):
    def __init__(self,
                 model: nn.Module,
                 patience: int,
                 direction: str = "min",
                 min_delta: float = 0.0001,
                 verbose: bool = False,
                 trainable_only: bool = False,
                 include_buffers: bool = True,
                 restore_best: bool = False):
        """Stop training if target metric does not improve.

        The model parameters with best performance are tracked and restored at the end.

        Parameters:
            model: Model to track.
            patience: How many iterations without improvement to tolerate. For example, patience=2 means that two
                      iterations *in a row* without improvement are okay; stopping would be triggered after the third
                      iteration in arow without improvement.
            direction: Whether the metric of interest is minimized (e.g. loss) or maximized (e.g. accuracy).
            min_delta: An improvement is only counted as such if it is better by at least this amount.
            verbose: If True, report on how things are going.
            trainable_only: See notes in base class.
            include_buffers: See notes in base class.
            restore_best: If True, when the stop signal is triggered, the best parameters are restored to the model.
                          Otherwise, you have to do this manually later.
        """
        super().__init__(model, trainable_only, include_buffers)
        if direction not in ["min", "max"]:
            raise ValueError(f"direction should be 'min' or 'max', you passed {direction}")
        self.best_value = np.inf if direction == "min" else -np.inf
        self.direction = direction
        self.min_delta = min_delta

        self.patience = patience
        self.disappointment = 0
        self.verbose = verbose
        if verbose and patience is None:
            print("EarlyStopping with patience None -- noop and will never stop")
        self.restore_best = restore_best

    def update(self,
               value: ScalarFloat) -> bool:
        """Run one 'iteration' of early stopping.

        This updates the patience parameter, and if stopping is triggered, sends a signal to stop training. This
        function does *not* actually stop the training process; this needs to be handled in the training function
        based on the bool this function returns. *Optionally* restores the best tracked model parameters.

        Parameters:
            value: New value to compare to best so far.
        """
        if self.patience is None:
            return False
            
        if ((self.direction == "min" and value < self.best_value - self.min_delta) 
            or (self.direction == "max" and value > self.best_value + self.min_delta)):
            self.best_value = value
            self.update_best()
            self.disappointment = 0
            if self.verbose:
                print("New best value found; no longer disappointed")
            return False
        else:
            self.disappointment += 1
            if self.verbose:
                print(f"EarlyStopping disappointment increased to {self.disappointment}")

            if self.disappointment > self.patience:
                if self.verbose:
                    print("EarlyStopping has become too disappointed; now would be a good time to cancel training")
                    if self.restore_best:
                        print("Restoring best model from state_dict")
                        self.apply_parameters()
                return True
            else:
                return False
            
    @torch.no_grad()
    def update_best(self):
        """Update saved state with new best."""
        for best_param, model_param in zip(self.tracked_parameters, self.get_parameters()):
            best_param[:] = model_param


class EMA(ParameterTracker):
    def __init__(self,
                 model: nn.Module,
                 momentum: float,
                 trainable_only: bool = False,
                 include_buffers: bool = True):
        """Exponential Moving Average for model parameters.
        
        After each training step, the average is updated with new parameters. That is assuming you call update() after
        each step. :)

        Parameters:
            model: Guess what...
            momentum: How fast/slow the update is. We use ema_new = momentum*ema_old + (1-momentum)*current_params.
            trainable_only: See notes in base class.
            include_buffers: See notes in base class.
        """
        super().__init__(model, trainable_only, include_buffers)
        self.momentum = momentum

    @torch.no_grad()
    def update(self):
        """Should be called once per training step, or however often you want to update the EMA."""
        for ema_param, model_param in zip(self.tracked_parameters, self.get_parameters()):
            ema_param.mul_(self.momentum).add_(model_param, alpha=1 - self.momentum)


class Checkpointer:
    def __init__(self,
                 model: nn.Module,
                 directory: str,
                 checkpoint_name: str,
                 frequency: int):
        """Regularly saves model weights (via state_dict) during training.
        
        Parameters:
            model: Model to store checkpoints for.
            directory: Path to store checkpoints to. Will be created if non-existent.
            checkpoint_name: Base name for each checkpoint file. Epoch indices will be appended.
            frequency: Will create a checkpoint every this many epochs.
        """
        self.model = model
        self.directory = directory
        self.checkpoint_name = checkpoint_name
        self.frequency = frequency
        if not os.path.exists(directory):
            os.makedirs(directory)

    def maybe_checkpoint(self,
                         epoch_ind: int):
        """Create a new checkpoint if the trigger has been met.
        
        Parameters:
            epoch_ind: Self-explanatory.
        """
        # could check epoch_ind > 0, but maybe someone wants to evaluate the untrained model...?
        if not epoch_ind % self.frequency:
            path = os.path.join(self.directory, self.checkpoint_name + f"_{epoch_ind:04}.pt")
            torch.save(self.model.state_dict(), path)


def cosine_decay_warmup(optimizer, n_steps_decay, n_steps_warmup):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/n_steps_warmup, total_iters=n_steps_warmup)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps_decay)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[n_steps_warmup])
