from datetime import datetime
from typing import Tuple


def check_milestone(v: str) -> str:
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")

    today = datetime.today().strftime("%Y-%m-%d")
    assert "2020-01-01" <= v <= today, f"{v} is not a valid milestone"
    return v


def check_epochs(v: int) -> int:
    assert 1 <= v <= 30, f"{v} is not a valid epochs"
    return v


def check_normalizer(v: str) -> str:
    assert v in ["zscore", "cs_zscore"], f"{v} is not a valid normalizer"
    return v


def check_range_tuple(v: Tuple[str, str]) -> Tuple[str, str]:
    if v is not None:
        assert len(v) == 2, f"{v} should be range tuple of length 2."
        assert v[0] <= v[1], f"{v} is not a valid range tuple: left < right"
    return v
