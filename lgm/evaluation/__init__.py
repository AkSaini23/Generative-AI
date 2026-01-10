"""This module provides metrics as well as the requires classifiers for evaluating generative models.

We provide inception score and FID, and classifiers for all datasets supported in lgm.data except flickr, celeba and
stl10.

REFERENCES
Inception Score: https://proceedings.neurips.cc/paper_files/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf
FID: https://papers.nips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html
Original Precision & Recall: https://arxiv.org/abs/1806.00035
Improved version (implemented here): https://research.nvidia.com/sites/default/files/pubs/2019-12_Improved-Precision-and/kynkaanniemi2019metric_paper.pdf
"""
from .metrics import fid, fid_reconstructions, inception_score, inception_score_with_features, precision_recall
from .networks import get_classifier, get_resnet
from .training import (accuracy, accuracy_sequence, augmentations, top_k_accuracy, top_k_accuracy_sequence,
                       ClassifierTrainer)
