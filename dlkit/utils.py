import collections
import datetime
from typing import Literal, List

__all__ = [
    "CHECHPOINT_META",
    "get_time_slots",
]

CheckpointMeta = collections.namedtuple(
    "CheckpointMeta",
    [
        "prefix_dir",
        "training_args",
        "normalizer",
        "model",
    ],
)

CHECHPOINT_META = CheckpointMeta(
    prefix_dir="checkpoint",
    training_args="training_args.yaml",
    normalizer="normalizer.pt",
    model="model.pt",
)

# # Name of the files used for checkpointing
# TRAINING_ARGS_NAME = "training_args.pt"
# NORMALIZER_NAME = "normalizer.pt"
# MODEL_NAME = "model.pt"
# PREFIX_CHECKPOINT_DIR = "checkpoint"


def get_time_slots(
    start: str = "0930",
    end: str = "1030",
    freq_in_min: Literal[1, 10] = 10,
    bar_on_the_right: bool = True,
) -> List[str]:
    """Generate the list of intraday time slots.

    Args:
        start (str, optional): start slot. Defaults to "0930".
        end (str, optional): end slot. Defaults to "1030".
        freq_in_min (Literal[1, 10], optional): number of minutes as stride. Defaults to 10.
        bar_on_the_right (bool, optional): start slot exclusive if True,
            otherwise end slot exclusive. Defaults to True.

    Returns:
        list[str]: list of time slots.

    Examples:
        note that trading time is 0930-1130, 1300-1500.
        >>> get_time_slots(start="0930", end="1030", freq_in_min=10, bar_on_the_right=True)
        ['0940', '0950', '1000', '1010', '1020', '1030']
        >>> get_time_slots(start="0930", end="1030", freq_in_min=10, bar_on_the_right=False)
        ['0930', '0940', '0950', '1000', '1010', '1020']
        >>> get_time_slots(start="1030", end="1300", freq_in_min=10, bar_on_the_right=True)
        ['1040', '1050', '1100', '1110', '1120', '1130']
    """
    _start = datetime.datetime.strptime(start, "%H%M")
    _end = datetime.datetime.strptime(end, "%H%M")

    # A-shares
    morning_start = datetime.datetime.strptime("0930", "%H%M")
    morning_end = datetime.datetime.strptime("1130", "%H%M")
    afternoon_start = datetime.datetime.strptime("1300", "%H%M")
    afternoon_end = datetime.datetime.strptime("1500", "%H%M")

    freq = datetime.timedelta(minutes=freq_in_min)
    slots = []
    while _start <= _end:
        slots.append(_start)
        _start += freq

    if bar_on_the_right is True:
        slots = slots[1:]
        slots = [
            _s
            for _s in slots
            if morning_start < _s <= morning_end
            or afternoon_start < _s <= afternoon_end
        ]
    else:
        slots = slots[:-1]
        slots = [
            _s
            for _s in slots
            if morning_start <= _s < morning_end
            or afternoon_start <= _s < afternoon_end
        ]

    # right end, so 0930, 1300 excluded.
    slots = [f"{_s:%H%M}" for _s in slots]
    return slots
