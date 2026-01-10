import os
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, inception_v3, Inception_V3_Weights
from torchvision.transforms.v2 import Normalize

from ..layers import CNNBody, ConvBlockPrenorm


def get_resnet(protocol: str = "lax"):
    """Get network for more general feature extraction.
    
    For datasets where we have no classifier (e.g. no labels), we need some kind of general feature extractor for
    typical evaluation metrics. This provides one such network. We could really use any ImageNet model, we just have to
    choose one. Use this one for comparison's sake!

    Parameters:
        protocol: One of 'lax' or 'strict'. 'lax' uses a Resnet50 because that's the first thing I used because I just
                  wanted to try something and now we are stuck with it to have values be internally comparable. :) But
                  to have values that should be comparable to literature, you can use 'strict' which uses an Inceptionv3
                  network instead.
    """
    if protocol == "lax":
        extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        upsample = 224
    elif protocol == "strict":
        extractor = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        upsample = 299
    else:
        raise ValueError(f"Unsupported protocol {protocol}. Allowed are 'lax', 'strict'.")
    
    extractor.fc = nn.Identity()
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    upsample_net = nn.Sequential(nn.Upsample(size=upsample, mode="bilinear"),
                                 normalizer,
                                 extractor)
    return upsample_net.eval()


def get_classifier(dataset: str,
                   weight_path: str | None = None) -> nn.Module:
    """Get a pre-built classifier architecture for various datasets.
    
    If you want to use our provided weights for FID/IS evaluation, you must build the architectures using these
    functions. Note that loading weights is optional; you could also train them yourself. But then your results will
    not be comparable to anybody else's!
    """
    if dataset == "mnist":
        classifier = mnist_classifier()
    elif dataset == "emnist":
        classifier = emnist_classifier()
    elif dataset == "fashion":
        classifier = fashion_classifier()
    elif dataset == "svhn":
        classifier = svhn_classifier()
    elif dataset == "cifar10":
        classifier = cifar10_classifier()
    elif dataset == "cifar100":
        classifier = cifar100_classifier()
    elif dataset == "speechcommands":
        classifier = speechcommands_classifier()
    else:
        raise ValueError("invalid dataset")
    
    if weight_path is not None:
        weight_path = os.path.join(weight_path, f"classifier_weights_{dataset}.pt")
        classifier.load_state_dict(torch.load(weight_path))
    return classifier


def mnist_classifier() -> nn.Module:
    level_filters = [32, 64, 128, 256]
    level_strides = [1, 2, 2, 2]
    blocks_per_level = 2
    kernel_size = 3
    activation = nn.Mish
    input_shape = (1, 32, 32)
    input_channels = input_shape[0]
    
    block_fn = ConvBlockPrenorm
    block_kwargs = {"n_layers": 2, "activation": activation, "kernel_size": kernel_size,
                    "use_se": True, "se_reduction_factor": 8, "se_bias": 1.,
                    "use_residual": True, "bottleneck_factor": 1}
    return cnn_classifier(input_channels, 10, level_filters, blocks_per_level, level_strides, block_fn,
                          block_kwargs, True, 1, 1024, 0.25)


def fashion_classifier() -> nn.Module:
    level_filters = [64, 128, 256, 512]
    level_strides = [1, 2, 2, 2]
    blocks_per_level = 2
    kernel_size = 3
    activation = nn.Mish
    input_shape = (1, 32, 32)
    input_channels = input_shape[0]
    
    block_fn = ConvBlockPrenorm
    block_kwargs = {"n_layers": 2, "activation": activation, "kernel_size": kernel_size,
                    "use_se": True, "se_reduction_factor": 8, "se_bias": 1.,
                    "use_residual": True, "bottleneck_factor": 1}
    return cnn_classifier(input_channels, 10, level_filters, blocks_per_level, level_strides, block_fn,
                          block_kwargs, True, 1, 1024, 0.25)


def emnist_classifier() -> nn.Module:
    level_filters = [64, 128, 256, 512]
    level_strides = [1, 2, 2, 2]
    blocks_per_level = 2
    kernel_size = 3
    activation = nn.Mish
    input_shape = (1, 32, 32)
    input_channels = input_shape[0]
    
    block_fn = ConvBlockPrenorm
    block_kwargs = {"n_layers": 2, "activation": activation, "kernel_size": kernel_size,
                    "use_se": True, "se_reduction_factor": 8, "se_bias": 1.,
                    "use_residual": True, "bottleneck_factor": 1}
    return cnn_classifier(input_channels, 47, level_filters, blocks_per_level, level_strides, block_fn,
                          block_kwargs, True, 1, 2048, 0.25)


def svhn_classifier() -> nn.Module:
    level_filters = [64, 128, 256, 512]
    level_strides = [1, 2, 2, 2]
    blocks_per_level = 3
    kernel_size = 3
    activation = nn.Mish
    input_shape = (3, 32, 32)
    input_channels = input_shape[0]
    
    block_fn = ConvBlockPrenorm
    block_kwargs = {"n_layers": 2, "activation": activation, "kernel_size": kernel_size,
                    "use_se": True, "se_reduction_factor": 8, "se_bias": 1.,
                    "use_residual": True, "bottleneck_factor": 1}
    return cnn_classifier(input_channels, 10, level_filters, blocks_per_level, level_strides, block_fn,
                          block_kwargs, True, 1, 1024, 0.25)


def cifar10_classifier() -> nn.Module:
    level_filters = [96, 192, 384, 768]
    level_strides = [1, 2, 2, 2]
    blocks_per_level = 3
    kernel_size = 3
    activation = nn.Mish
    input_shape = (3, 32, 32)
    input_channels = input_shape[0]
    
    block_fn = ConvBlockPrenorm
    block_kwargs = {"n_layers": 2, "activation": activation, "kernel_size": kernel_size,
                    "use_se": True, "se_reduction_factor": 8, "se_bias": 1.,
                    "use_residual": True, "bottleneck_factor": 1}
    return cnn_classifier(input_channels, 10, level_filters, blocks_per_level, level_strides, block_fn,
                          block_kwargs, True, 1, 2048, 0.25)


def cifar100_classifier() -> nn.Module:
    level_filters = [128, 256, 512, 1024]
    level_strides = [1, 2, 2, 2]
    blocks_per_level = 3
    kernel_size = 3
    activation = nn.Mish
    input_shape = (3, 32, 32)
    input_channels = input_shape[0]
    
    block_fn = ConvBlockPrenorm
    block_kwargs = {"n_layers": 2, "activation": activation, "kernel_size": kernel_size,
                    "use_se": True, "se_reduction_factor": 8, "se_bias": 1.,
                    "use_residual": True, "bottleneck_factor": 1}
    return cnn_classifier(input_channels, 100, level_filters, blocks_per_level, level_strides, block_fn,
                          block_kwargs, True, 1, 2048, 0.25)


def cnn_classifier(input_channels: int,
                   n_outputs: int,
                   level_filters: list[int],
                   blocks_per_level: int,
                   level_strides: list[int],
                   block_fn: type[nn.Module],
                   block_kwargs: dict[str, Any],
                   global_pool: bool,
                   n_linear: int,
                   linear_size: int,
                   dropout: float) -> nn.Module:
    classifier = nn.Sequential()
    classifier.append(nn.Conv2d(input_channels, level_filters[0], block_kwargs["kernel_size"], padding="same"))
    classifier.append(CNNBody(blocks_per_level, level_filters, level_strides,
                              input_channels=level_filters[0],
                              direction="down", block_fn=block_fn, **block_kwargs))
    if global_pool:
        classifier.append(nn.AdaptiveAvgPool2d(1))
    classifier.append(nn.Flatten())
    if dropout:
        classifier.append(nn.Dropout(dropout))
    for _ in range(n_linear):
        classifier.append(nn.LazyLinear(linear_size))
        classifier.append(block_kwargs["activation"]())
        if dropout:
            classifier.append(nn.Dropout(dropout))
    classifier.append(nn.LazyLinear(n_outputs))
    classifier.eval()
    return classifier


def speechcommands_classifier() -> nn.Module:
    level_filters = [32, 64, 128, 256, 512]
    level_strides = [1] +  [4] * (len(level_filters) - 1)
    blocks_per_level = 2
    kernel_size = 7
    activation = nn.Mish
    input_shape = (1, 16384)
    input_channels = input_shape[0]

    n_linear = 1
    linear_size = 1024
    n_outputs = 35
    
    block_fn = ConvBlockPrenorm
    block_kwargs = {"n_layers": 2, "activation": activation, "kernel_size": kernel_size, "drop_probability_max": 0.25,
                    "use_se": True, "se_reduction_factor": 8, "se_bias": 1., "data_format": "1d",
                    "use_residual": True, "bottleneck_factor": 1}
    
    classifier = nn.Sequential()
    classifier.append(nn.Conv1d(input_channels, level_filters[0], block_kwargs["kernel_size"], padding="same"))
    classifier.append(CNNBody(blocks_per_level, level_filters, level_strides,
                              input_channels=level_filters[0],
                              direction="down", block_fn=block_fn, **block_kwargs))
    classifier.append(nn.AdaptiveAvgPool1d(1))
    classifier.append(nn.Flatten())
    classifier.append(nn.Dropout(0.25))
    for _ in range(n_linear):
        classifier.append(nn.LazyLinear(linear_size))
        classifier.append(block_kwargs["activation"]())
        classifier.append(nn.Dropout(0.25))
    classifier.append(nn.LazyLinear(n_outputs))
    classifier.eval()
    return classifier
