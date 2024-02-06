from typing import Union, List, ParamSpec, TypeVar, Callable, Concatenate
from pathlib import Path

import polars as pl

P = ParamSpec("P")
T = TypeVar("T")

__all__ = ["ParquetStockSource"]


class ParquetStockSource:
    def __init__(
        self,
        data_path: Union[str, Path, List[str], List[Path]],
        function: Callable[Concatenate[pl.LazyFrame, P], pl.LazyFrame],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        self.df_lazy = pl.scan_parquet(data_path).pipe(function, *args, **kwargs)

    def __call__(self) -> pl.DataFrame:
        return self.df_lazy.collect()
