import pytest

import polars as pl
import datetime

from dlkit.preprocessing import CrossSectionalScaler, StandardScaler


@pytest.fixture
def generate_df():
    df_train = pl.DataFrame(
        {
            "date": pl.datetime_range(
                datetime.datetime(2020, 10, 10),
                datetime.datetime(2021, 12, 1),
                interval="1d",
                eager=True,
            )[:100],
            "symbol": [f"s{i}" for i in range(100)],
            "x1": pl.arange(0, 100, 1, eager=True) / 199.0,
            "x2": pl.arange(10, 110, 1, eager=True) / 19.0,
            "y2": pl.arange(110, 210, 1, eager=True) / 90.0,
        }
    )
    df_test = pl.DataFrame(
        {
            "date": pl.datetime_range(
                datetime.datetime(2021, 10, 10),
                datetime.datetime(2022, 12, 1),
                interval="1d",
                eager=True,
            )[:100],
            "symbol": [f"ss{i}" for i in range(10, 110)],
            "x1": pl.arange(11, 111, 1, eager=True) / 9.0,
            "x2": pl.arange(16, 116, 1, eager=True) / -19.0,
            # 'y2': pl.arange(155, 255, 1).truediv(9.9),
        }
    )
    return df_train, df_test


def test_normalizers(generate_df):  # noqa
    df_train, df_test = generate_df
    cs_normalizer = CrossSectionalScaler()
    cs_normalizer.fit_transform(df_train)
    cs_test = cs_normalizer.transform(df_test)

    assert cs_test.shape == (100, 4)

    z_normalizer = StandardScaler()
    z_normalizer.fit_transform(df_train)
    z_test = z_normalizer.transform(df_test)

    assert z_test.shape == (100, 4)
