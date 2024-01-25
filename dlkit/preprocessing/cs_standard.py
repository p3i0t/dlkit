import polars as pl
from dlkit.preprocessing.bases import DataFrameNormalizer


class CrossSectionalScaler(DataFrameNormalizer):
    required_columns = ["date", "symbol"]

    def __init__(self):
        ...

    def reset(self):
        if hasattr(self, "columns"):
            delattr(self, "columns")

    def __repr__(self):
        return f"CrossSectionalScaler(" f"columns={getattr(self, 'columns', None)})"

    def partial_fit(self, x: pl.DataFrame):
        if not set(self.required_columns).issubset(set(x.columns)):
            raise ValueError(
                f"required columns: {self.required_columns} missing in input dataframe."
            )
        columns = x.select(
            pl.col(pl.NUMERIC_DTYPES).exclude(self.required_columns)
        ).columns

        setattr(self, "n_samples_seen", len(x))
        setattr(self, "columns", columns)
        return self

    def fit(self, x: pl.DataFrame):
        self.reset()
        return self.partial_fit(x=x)

    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        """Normalize all numerical columns of input dataframe, except for the required columns.

        Args:
            x (pl.DataFrame): input dataframe.

        Raises:
            ValueError: required columns missing in input dataframe.
            ValueError: input dataframe contains unseen columns for this normalizer.

        Returns:
            pl.DataFrame: output dataframe.
        """
        if not set(self.required_columns).issubset(set(x.columns)):
            raise ValueError(
                f"required columns: {self.required_columns} missing in input dataframe."
            )
        columns = x.select(
            pl.col(pl.NUMERIC_DTYPES).exclude(self.required_columns)
        ).columns
        columns_diff = set(columns) - set(getattr(self, "columns"))
        if len(columns_diff) > 0:
            raise ValueError(
                f"{columns_diff} are extra columns not fitted by this normalizer."
            )

        x = x.with_columns(
            pl.col(c).sub(pl.col(c).mean()).truediv(pl.col(c).std()).over("date")
            for c in columns
        )
        return x.select(self.required_columns + columns)

    def fit_transform(self, x: pl.DataFrame) -> pl.DataFrame:
        return self.fit(x).transform(x)
