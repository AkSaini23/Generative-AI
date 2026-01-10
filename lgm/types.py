"""This module uses jaxtyping to add various more specific tensor types."""
from typing import TypeAlias, TypeVar, Union

from jaxtyping import Float, Int
from torch import Tensor


ImageFloat: TypeAlias = Float[Tensor, "c h w"]
ImageBatchFloat: TypeAlias = Float[Tensor, "batch c h w"]
BroadcastImageBatchFloat: TypeAlias = Float[Tensor, "batch c 1 1"]

SequenceFloat: TypeAlias = Float[Tensor, "c t"]
SequenceBatchFloat: TypeAlias = Float[Tensor, "batch c t"]
BroadcastSequenceBatchFloat: TypeAlias = Float[Tensor, "batch c 1"]

TabularFloat: TypeAlias = Float[Tensor, "c"]
TabularBatchFloat: TypeAlias = Float[Tensor, "batch c"]

DataFloat = TypeVar("DataFloat", ImageFloat, SequenceFloat, TabularFloat)
DataBatchFloat = TypeVar("DataBatchFloat", ImageBatchFloat, SequenceBatchFloat, TabularBatchFloat)
BroadcastDataBatchFloat = TypeVar("BroadcastDataBatchFloat", BroadcastImageBatchFloat, BroadcastSequenceBatchFloat)

ScalarFloat: TypeAlias = Float[Tensor, ""]
VectorBatchFloat: TypeAlias = Float[Tensor, "batch"]
BatchOrScalarFloat = TypeVar("BatchOrScalarFloat", ScalarFloat, VectorBatchFloat)

LabelBatchFloat = Union[DataBatchFloat, SequenceBatchFloat, TabularBatchFloat, VectorBatchFloat]
LabelFloat = Union[ImageFloat, SequenceFloat, TabularFloat, ScalarFloat]

AnyBatchFloat = LabelBatchFloat
AnyFloat = LabelFloat

IndexImageBatch: TypeAlias = Int[Tensor, "batch h w"]
IndexSequenceBatch: TypeAlias = Int[Tensor, "batch t"]
IndexBatch: TypeAlias = Int[Tensor, "batch"]
IndexDataBatch = TypeVar("IndexDataBatch", IndexImageBatch, IndexSequenceBatch, IndexBatch)

SquareMatrixFloat: TypeAlias = Float[Tensor, "b b"]


ImageFloatChannelsLast: TypeAlias = Float[Tensor, "h w c"]
ImageBatchFloatChannelsLast: TypeAlias = Float[Tensor, "batch h w c"]
BroadcastImageBatchFloatChannelsLast: TypeAlias = Float[Tensor, "batch 1 1 c"]

SequenceFloatChannelsLast: TypeAlias = Float[Tensor, "t c"]
SequenceBatchFloatChannelsLast: TypeAlias = Float[Tensor, "batch t c"]
BroadcastSequenceBatchFloatChannelsLast: TypeAlias = Float[Tensor, "batch 1 c"]

DataFloatChannelsLast = TypeVar("DataFloat", ImageFloatChannelsLast, SequenceFloatChannelsLast, TabularFloat)
DataBatchFloatChannelsLast = TypeVar("DataBatchFloat", ImageBatchFloatChannelsLast, SequenceBatchFloatChannelsLast,
                                     TabularBatchFloat)
BroadcastDataBatchFloatChannelsLast = TypeVar("BroadcastDataBatchFloat", BroadcastImageBatchFloatChannelsLast,
                                              BroadcastSequenceBatchFloatChannelsLast)
