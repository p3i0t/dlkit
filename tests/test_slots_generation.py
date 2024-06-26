import pytest

from dlkit.utils import get_time_slots


@pytest.mark.parametrize(
    "start, end, freq_in_min, bar_on_the_right, expected",
    [
        ("0930", "1030", 10, True, ["0940", "0950", "1000", "1010", "1020", "1030"]),
        (
            "0930",
            "1130",
            10,
            True,
            [
                "0940",
                "0950",
                "1000",
                "1010",
                "1020",
                "1030",
                "1040",
                "1050",
                "1100",
                "1110",
                "1120",
                "1130",
            ],
        ),
        ("1030", "1300", 10, True, ["1040", "1050", "1100", "1110", "1120", "1130"]),
        ("0930", "1030", 10, False, ["0930", "0940", "0950", "1000", "1010", "1020"]),
        (
            "0930",
            "1130",
            10,
            False,
            [
                "0930",
                "0940",
                "0950",
                "1000",
                "1010",
                "1020",
                "1030",
                "1040",
                "1050",
                "1100",
                "1110",
                "1120",
            ],
        ),
        ("1030", "1300", 10, False, ["1030", "1040", "1050", "1100", "1110", "1120"]),
        (
            "1030",
            "1400",
            10,
            True,
            [
                "1040",
                "1050",
                "1100",
                "1110",
                "1120",
                "1130",
                "1310",
                "1320",
                "1330",
                "1340",
                "1350",
                "1400",
            ],
        ),
        (
            "1030",
            "1400",
            10,
            False,
            [
                "1030",
                "1040",
                "1050",
                "1100",
                "1110",
                "1120",
                "1300",
                "1310",
                "1320",
                "1330",
                "1340",
                "1350",
            ],
        ),
    ],
)
def test_get_time_slots(start, end, freq_in_min, bar_on_the_right, expected):
    o = get_time_slots(start, end, freq_in_min, bar_on_the_right)
    print(f"{o=}, {expected=}")
    assert expected == o
