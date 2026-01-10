"""This module contains various reusable layers or amalgamations of layers to build architectures with.

REFERENCES
ResNet: https://arxiv.org/abs/1512.03385
Preactivation/normalization blocks: https://arxiv.org/abs/1603.05027
Squeeze-and-excite: https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
Batch normalization: https://arxiv.org/abs/1502.03167
Group normalization: https://arxiv.org/abs/1803.08494

Some activation functions:
ReLU: https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf
PReLU (and He/Kaiming initialization): https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
GELU: https://arxiv.org/abs/1606.08415
Swish/SiLU: https://arxiv.org/abs/1710.05941v1
Mish (my favorite :)): https://arxiv.org/abs/1908.08681
"""
from .conv import CNNBody, CNNLevel, ConvBlockPrenorm, DownsampleAA, SqueezeExcite, UpsampleAA
from .generic import hidden_linear, OneHot, Reshape
from .norms import BNWrap1d, BNWrap2d, DummyNorm, LNWrap
