import pytest
import polars as pl

from dlkit.data.data_source import ParquetStockSource


@pytest.fixture
def generate_parquet_files():
    from datetime import datetime
    from pathlib import Path

    df1 = pl.DataFrame(
        {
            "a": pl.arange(0, 100, 1, eager=True),
            "datetime": pl.datetime_range(
                datetime(2022, 10, 1), datetime(2023, 12, 1), interval="1d", eager=True
            )[:100],
        }
    )
    df2 = pl.DataFrame(
        {
            "a": pl.arange(10, 110, 1, eager=True),
            "datetime": pl.datetime_range(
                datetime(2020, 10, 1), datetime(2021, 12, 1), interval="1d", eager=True
            )[:100],
        }
    )

    parq_dir = Path("parq_dir")
    parq_dir.mkdir(parents=True, exist_ok=True)
    df1.write_parquet(f"{parq_dir}/df1.parq")
    df2.write_parquet(f"{parq_dir}/df2.parq")

    return parq_dir


def test_parq_data_source(generate_parquet_files):  # noqa
    def foo_pow(df: pl.LazyFrame, col: str):
        return df.with_columns(pl.col(col).pow(2).alias("a^2"))

    parq_dir = generate_parquet_files
    ds = ParquetStockSource(
        # data_path=[f"{parq_dir}/df1.parq", f"{parq_dir}/df2.parq"],
        data_path=f"{parq_dir}/*",
        function=foo_pow,
        col="a",
    )
    data = ds()
    # print(f"{data.head()=}")
    assert data.shape == (200, 3)
