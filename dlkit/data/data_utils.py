"""This module contains the data utilities for the pit package.
"""
from dataclasses import dataclass
from typing import Literal, NamedTuple, Sequence, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import torch

# from pydantic import BaseModel, Field, PlainValidator
from typing_extensions import Annotated

ArrayType: TypeAlias = Union[npt.NDArray[np.float32], torch.FloatTensor]
MonitorMode: TypeAlias = Literal["min", "max"]
# SeriesType: TypeAlias = Union[Sequence[object], Series]
# NormalizerScheme: TypeAlias = Literal["cross-sectional", "standard", "none"]

ArrayStr = npt.NDArray[np.str_]


def check_1d_np_array(v: np.ndarray) -> np.ndarray:
    """pydantic validator for 1d np.ndarray.

    Args:
        v (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    assert v.ndim == 1, "Array must be 1d."
    return v


def check_3d_np_array(v: np.ndarray) -> np.ndarray:
    """pydantic validator for 3d np.ndarray.

    Args:
        v (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    assert v.ndim == 3, "Array must be 3d."
    return v


# def check_float32_array(v: ArrayType) -> ArrayType:
#     if isinstance(v, torch.Tensor):
#         assert v.dtype == torch.float32, "Array must be float32."
#     elif isinstance(v, np.ndarray):
#         assert v.dtype == np.float32, "Array must be float32."
#     else:
#         raise TypeError("Array must be torch.Tensor or np.ndarray.")
#     return v


def check_2d_or_3d_float32_tensor(v: torch.Tensor) -> torch.Tensor:
    """pydantic validator for 2d or 3d float32 torch.Tensor.

    Args:
        v (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    assert v.dtype == torch.float32, "Array must be float32."
    assert v.ndim in [2, 3], "Array must be 2d or 3d."
    return v


@dataclass
class StockDataset:
    date: np.ndarray
    symbol: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_columns: Sequence[str]

    @property
    def x_shape(self):
        return self.x.shape[1:]

    @property
    def y_shape(self):
        return self.y.shape[1:]

    def __len__(self):
        return len(self.date)


# class StockDataset(BaseModel):
#     date: Annotated[np.ndarray, PlainValidator(check_1d_np_array)] = Field(
#         ..., description="date of the samples"
#     )
#     symbol: Annotated[np.ndarray, PlainValidator(check_1d_np_array)] = Field(
#         ..., description="symbol of the samples"
#     )
#     x: Annotated[
#         np.ndarray,
#         PlainValidator(check_3d_np_array),
#     ] = Field(..., description="features of the samples, 3D array")
#     y: Annotated[
#         np.ndarray,
#         PlainValidator(check_3d_np_array),
#     ] = Field(..., description="labels of the samples, 3D array")
#     y_columns: Sequence[str] = Field(..., description="column names of y")

#     @property
#     def x_shape(self):
#         return self.x.shape[1:]

#     @property
#     def y_shape(self):
#         return self.y.shape[1:]

#     def __len__(self):
#         return len(self.date)


@dataclass
class StockBatch:
    date: np.ndarray
    symbol: np.ndarray
    x: np.ndarray | torch.Tensor
    y: np.ndarray | torch.Tensor
    y_columns: Sequence[str]

    @property
    def x_shape(self):
        return self.x.shape[1:]

    @property
    def y_shape(self):
        return self.y.shape[1:]

    def __len__(self):
        return len(self.date)
    
    def __repr__(self) -> str:
        return f"StockBatch(date={self.date.shape, type(self.date)}, symbol={self.symbol.shape, type(self.symbol)}, x={self.x.shape, type(self.x)}, y={self.y.shape, type(self.y)}, y_columns={self.y_columns})"


# class StockBatch(BaseModel):
#     """Class for a typical stock batch, including date, symbol, x, y, y_columns etc.

#     Args:
#         BaseModel (_type_): _description_

#     Raises:
#         AttributeError: _description_

#     Returns:
#         _type_: _description_
#     """

#     date: Annotated[np.ndarray, PlainValidator(check_1d_np_array)] = Field(
#         ..., description="date of the samples"
#     )
#     symbol: Annotated[np.ndarray, PlainValidator(check_1d_np_array)] = Field(
#         ..., description="symbol of the samples"
#     )
#     x: Annotated[
#         torch.Tensor,
#         PlainValidator(check_2d_or_3d_float32_tensor),
#     ] = Field(..., description="features of the samples, 2D or 3D array")
#     y: Annotated[
#         torch.Tensor,
#         PlainValidator(check_2d_or_3d_float32_tensor),
#     ] = Field(None, description="labels of the samples, 2D or 3D array")
#     y_columns: Sequence[str] = Field(..., description="column names of y")

#     @property
#     def x_shape(self):
#         return self.x.shape[1:]

#     @property
#     def y_shape(self):
#         if hasattr(self, "y"):
#             return self.y.shape[1:]
#         else:
#             raise AttributeError("StockBatch has no attribute 'y'.")

#     def __len__(self):
#         return len(self.date)


StockCrossSectionalBatch: TypeAlias = StockBatch


class EvalPrediction(NamedTuple):
    date: ArrayStr
    symbol: ArrayStr
    pred: ArrayType
    y: ArrayType
    y_columns: list[str]


def cal_n_parameters(model: torch.nn.Module):
    """Calculate the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NumpyStockDataLoader:
    def __init__(
        self,
        dataset: StockDataset,
        shuffle: bool = False,
        batch_size: int = 256,
        drop_last: bool = False,
        # device: torch.device | str = torch.device("cuda"),
    ):
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        # self.device = device
        self.indices = np.arange(len(self.dataset))
        n_batches, remainder = divmod(len(self.dataset), batch_size)
        self.n_batches = n_batches + (1 if remainder > 0 and drop_last is False else 0)

    def __iter__(self):
        self.iter = 0
        if self.shuffle:
            self.indices = self.indices[np.random.permutation(len(self.dataset))]
        return self

    def __next__(self) -> StockBatch:
        if self.iter >= self.n_batches:
            raise StopIteration
        start = self.iter * self.batch_size
        end = (
            (self.iter + 1) * self.batch_size
            if self.iter < self.n_batches - 1
            else len(self.dataset)
        )
        self.iter += 1
        # all numpy arrays
        _i = self.indices[start:end]
        _date = self.dataset.date[_i]
        _symbol = self.dataset.symbol[_i]
        _x = self.dataset.x[_i]
        _y = self.dataset.y[_i]
        batch = StockBatch(
            date=_date,
            symbol=_symbol,
            x=_x,
            y=_y,
            y_columns=self.dataset.y_columns,
        )
        return batch

    def __len__(self):
        return self.n_batches
