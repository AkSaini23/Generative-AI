from collections.abc import Callable

import numpy as np
import scipy.linalg
import torch
from torch import nn
from tqdm.auto import tqdm

from ..common import squared_distances
from ..types import DataBatchFloat, LabelBatchFloat, TabularBatchFloat


def fid(feature_extractor: nn.Module,
        dataloader: torch.utils.data.DataLoader[tuple[DataBatchFloat, LabelBatchFloat]],
        generate_fn: Callable[[int], DataBatchFloat],
        n_samples: int | None = None,
        return_features: bool = False,
        use_tqdm: bool = False) -> float | tuple[float, np.ndarray, np.ndarray]:
    """Computes FID.
    

    Parameters:
        feature_extractor: A module acting as the feature extractor to compute FID on. Usually a classifier minus the
                           "classification head".
        dataloader: Should provide the real data to compare to.
        generate_fn: Function that accepts an n_samples argument and generates as many data points. Usually a model's
                     generate() function, or a thin wrapper around it.
        n_samples: How many samples to generate in total. If not given, generate as many as there are real data points
                   in the dataloader.
        return_features: If True, returns not just the FID score, but also arrays containing the real and generated
                         features. This can be useful if you want to compute the Inception Score right afterwards. Then
                         you don't have to generate a new set of samples.
        use_tqdm: If True, displays a progressbar.
    """
    feature_extractor.eval()
    real_features = []
    device = next(feature_extractor.parameters()).device
    with torch.inference_mode():
        # manual progressbar required due to multiprocessing in dataloaders
        with tqdm(total=len(dataloader), desc="Computing real features", disable=not use_tqdm) as progressbar:
            for batch, _ in dataloader:
                real_feature_batch = feature_extractor(batch.to(device))
                real_features.append(real_feature_batch.cpu().numpy())
                progressbar.update(1)
    real_features = np.concat(real_features)
    
    if n_samples is None:
        batches_to_gen = int(np.ceil(real_features.shape[0] / dataloader.batch_size))
    else:
        batches_to_gen = int(np.ceil(n_samples / dataloader.batch_size))
    generated_features = []
    with torch.inference_mode():
        for _ in tqdm(range(batches_to_gen), desc="Computing generated features", disable=not use_tqdm):
            generated = generate_fn(dataloader.batch_size)
            gen_feature_batch = feature_extractor(generated)
            generated_features.append(gen_feature_batch.cpu().numpy())
    generated_features = np.concat(generated_features)

    fid_score = fid_with_features(real_features, generated_features)
    if return_features:
        return fid_score, real_features, generated_features
    else:
        return fid_score
    

def fid_reconstructions(feature_extractor: nn.Module,
                        reconstructor: nn.Module,
                        dataloader: torch.utils.data.DataLoader[tuple[DataBatchFloat, LabelBatchFloat]],
                        return_features: bool = False,
                        use_tqdm: bool = False) -> float | tuple[float, np.ndarray, np.ndarray]:
    """Compute FID, but for reconstructions rather than pure generations.
    
    This provides a unified measure to compare, for example, VAEs with auxiliary GAN losses and ones that don't have it.
    The former will usually have worse reconstruction losses, even if their outputs may subjectively look better. As
    such.

    Parameters:
        reconstructor: Model that takes a batch of data and returns reconstructions.
        Other parameters: Like fid().
    """
    feature_extractor.eval()
    reconstructor.eval()
    real_features = []
    reconstructed_features = []
    device = next(feature_extractor.parameters()).device
    with torch.inference_mode():
        # manual progressbar required due to multiprocessing in dataloaders
        with tqdm(total=len(dataloader), desc="Computing real & reconstructed features",
                  disable=not use_tqdm) as progressbar:
            for batch, _ in dataloader:
                batch = batch.to(device)
                real_feature_batch = feature_extractor(batch)
                real_features.append(real_feature_batch.cpu().numpy())

                reconstructed = reconstructor(batch)
                reconstructed_feature_batch = feature_extractor(reconstructed)
                reconstructed_features.append(reconstructed_feature_batch.cpu().numpy())
                progressbar.update(1)
    real_features = np.concat(real_features)
    reconstructed_features = np.concat(reconstructed_features)
    fid_score = fid_with_features(real_features, reconstructed_features)
    if return_features:
        return fid_score, real_features, reconstructed_features
    else:
        return fid_score


def fid_with_features(real_features: np.ndarray,
                      generated_features: np.ndarray) -> float:
    """Core FID computation from https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""
    # maybe float64 increases precision?
    real_features = real_features.astype(np.float64)
    generated_features = generated_features.astype(np.float64)
    mu1 = real_features.mean(axis=0)
    mu2 = generated_features.mean(axis=0)
    diff = mu1 - mu2
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(generated_features, rowvar=False)
    eps = 1e-6

    # product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    fid_score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid_score


def inception_score(classifier: nn.Module,
                    generate_fn: Callable[[int], DataBatchFloat],
                    n_samples: int,
                    batch_size: int,
                    use_tqdm: bool = False) -> float:
    """Computes the Inception Score for evaluation.
    
    Note that this has somewhat fallen out of favor; when in doubt, prefer the FID.

    Parameters:
        classifier: A classifier network for the dataset in question.
        generate_fn: See notes in the fid function.
        n_samples: Total number of samples to generate. Other than for FID, we dot not have real data here, so this
                   must always be given.
        batch_size: How many samples to generate at noce.
        use_tqdm: Whether to display a progressbar.
    """
    classifier.eval()
    batches_to_gen = int(np.ceil(n_samples / batch_size))
    log_probabilities = []
    with torch.inference_mode():
        for _ in tqdm(range(batches_to_gen), desc="Computing generated classifications", disable=not use_tqdm):
            generated = generate_fn(batch_size)
            gen_output_batch = torch.nn.functional.log_softmax(classifier(generated), dim=1)
            log_probabilities.append(gen_output_batch.cpu().numpy())
    return inception_base(log_probabilities)


def inception_score_with_features(classifier_head: nn.Module,
                                  generated_features: np.ndarray,
                                  batch_size: int,
                                  use_tqdm: bool = False) -> float:
    """Computes the Inception Score, but from pre-computed features.
    
    Useful if you compute the FID beforehand. Do that with return_features=True and then put the generated features in
    here.

    Parameters:
        classifier_head: Shallow classifier network based on pre-computed features. Should be whatever you left off the
                         feature extractor for FID. For example, if you did model[:-k] for the extractor, the head
                         should be model[-k:].
        gen_features: Array of features for generated data as returned by fid.
        batch_size: Batch size for what we put into the classifier head.
        use_tqdm: Show progressbar or not.
    """
    device = next(classifier_head.parameters()).device
    n_batches = int(np.ceil(len(generated_features) / batch_size))
    log_probabilities = []
    with torch.inference_mode():
        for batch_ind in tqdm(range(n_batches), desc="Computing generated classifications", disable=not use_tqdm):
            start_ind = batch_ind * batch_size
            batch = generated_features[start_ind:start_ind + batch_size]
            gen_output_batch = torch.nn.functional.log_softmax(classifier_head(torch.tensor(batch).to(device)), dim=1)
            log_probabilities.append(gen_output_batch.cpu().numpy())
    return inception_base(log_probabilities)


def inception_base(log_probabilities: list[np.ndarray]) -> float:
    """Core logic for the Inception Score.
    
    Parameters:
        log_probabilities: Should be an array of log-probs returned from a classifier for a set of generated data.
    """
    log_probabilities = np.concat(log_probabilities).astype(np.float64)  # prefer float64 for precision
    probabilities = np.exp(log_probabilities)
    mean_probs = probabilities.mean(axis=0, keepdims=True)
    klds = (probabilities * (log_probabilities - np.log(mean_probs))).sum(axis=1)
    return np.exp(klds.mean())


def precision_recall(real_features: np.ndarray,
                     generated_features: np.ndarray,
                     batch_size: int,
                     k: int,
                     use_tqdm: bool = False) -> dict[str, torch.Tensor]:
    """Computes precision, recall and their harmonic mean (F-Score) for generative models.
    
    Currently only a feature-based implementation. You can compute FID with return_features=True and then put those in
    here. 
    This method is based on approximate manifolds estimated via k-nearest neighbors. Since k is somewhat arbitrary, we
    compute results for all values up to the specified k. You then have two options, basically:
    - Choose one k and compare all models with respect to this k, always.
    - Compare the entire curves in some way, or average over the different values.
    The original paper proposes simply using k=3.
    """
    real_features = torch.tensor(real_features)
    generated_features = torch.tensor(generated_features)
    precision = manifold_compare(real_features, generated_features, batch_size, k, use_tqdm)
    recall = manifold_compare(generated_features, real_features, batch_size, k, use_tqdm)
    fscore = 2*precision*recall / (precision + recall)
    return {"precisions": precision, "recalls": recall, "fscores": fscore}


def manifold_compare(truth_features: TabularBatchFloat,
                        compare_features: TabularBatchFloat,
                        batch_size: int,
                        k: int,
                        use_tqdm: bool = False) -> torch.Tensor:
    """Estimates overlap between two distributions.
    
    The "real" manifold is first estimated. Then, for each sample in the distribution we want to compare to the target
    one, we check the istances to all samples in the real distribution, and for each real one, compare that to how
    close the compared sample is to other samples in the real distribution. If the distance between compared and real
    sample is lower than the k-th smallest distance between the real sample and other real ones, we count a match. At
    the end, we return the proportion of matches. This has two special cases:
    - Target real data, compare generated: Computes precision.
    - Target generated data, compare real: Computes recall.
    """
    truth_manifold = manifold_estimate(truth_features, batch_size, k, use_tqdm)

    def summary_fn(distances):
        full_comparison = distances[:, :, torch.newaxis] <= truth_manifold
        in_manifold = torch.any(full_comparison, dim=1).to(torch.float32)
        return in_manifold

    in_manifold = batched_distances_summary(compare_features, truth_features, batch_size, summary_fn, k,
                                            "Comparing manifold" if use_tqdm else "")
    in_count = in_manifold.sum(dim=0)
    return in_count / compare_features.shape[0]


def manifold_estimate(features: TabularBatchFloat,
                         batch_size: int,
                         k: int,
                         use_tqdm: bool = False) -> TabularBatchFloat:
    """Estimates manifold of a distribution represented by features via k-nearest neighbors.
    
    This computes the distance between each feature vector and all other feature vectors, and recorsd the k smallest
    distances for each. This estimates the manifold by showing how close samples are to each other.
    """
    def summary_fn(distances):
        return torch.topk(distances, k + 1, largest=False).values[:, 1:]
    
    return batched_distances_summary(features, features, batch_size, summary_fn, k,
                                     "Approximating manifold" if use_tqdm else "")


def batched_distances_summary(row_features: TabularBatchFloat,
                              column_features: TabularBatchFloat,
                              batch_size: int,
                              summary_fn,
                              k: int,
                              tqdm_description: str) -> TabularBatchFloat:
    """Compute distances between all points in two sets of features. and summarizes them in some way.

    This is needed for both manifold functions above, so we wrote a unified function. This makes the code harder to
    understand than a more explicit implementation, but significantly reduced code duplication.
    """
    n_row_batches = int(np.ceil(len(row_features) / batch_size))
    n_col_batches = int(np.ceil(len(column_features) / batch_size))
    summarized = torch.zeros(row_features.shape[0], k)

    for row_batch_ind in tqdm(range(n_row_batches), desc=tqdm_description, disable=not tqdm_description):
        row_start_ind = row_batch_ind * batch_size
        row_batch = row_features[row_start_ind:row_start_ind + batch_size]
        row_distances = torch.zeros(row_batch.shape[0], column_features.shape[0])
        
        for col_batch_ind in range(n_col_batches):
            col_start_ind = col_batch_ind * batch_size
            col_batch = column_features[col_start_ind:col_start_ind + batch_size]
            batch_dists = squared_distances(row_batch, col_batch)
            row_distances[:, col_start_ind:col_start_ind + batch_size] = batch_dists
        summarized[row_start_ind:row_start_ind + batch_size] = summary_fn(row_distances)
    return summarized
