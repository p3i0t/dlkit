import datetime
import itertools
import shutil

import pytest
import polars as pl
import numpy as np

from dlkit.train import TrainArguments
from dlkit.data.data_utils import StockDataset


@pytest.fixture
def generate_trainarguments() -> TrainArguments:
    """Fixture for generating TrainArguments.

    Returns:
        TrainArguments: fixture.
    """
    from pathlib import Path
    import torch
    milestone = "2022-07-11"
    SAVE_DIR = Path("tests/save")
    DATA_DIR = Path("tests/data")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import uuid

    d_in = 30
    d_out = 2
    x_columns = [str(uuid.uuid4()) for _ in range(d_in)]
    y_columns = [f"ret{i}" for i in range(d_out)]

    return TrainArguments(
        prod="1030",
        universe="euniv_largemid",
        save_dir=SAVE_DIR,
        dataset_dir=DATA_DIR,
        milestone=milestone,
        x_columns=x_columns,
        x_begin="0930",
        x_end="1030",
        freq_in_min=10,
        y_columns=y_columns,
        y_slots=['1020', "1030"],
        train_date_range=("2018-01-01", "2022-06-11"),
        eval_date_range=("2022-06-12", "2022-07-11"),
        test_date_range=("2022-07-12", "2022-08-12"),
        normalizer="zscore",
        monitor_metric="loss",
        monitor_mode="min",
        model="GPT_small",
        epochs=2,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_batch_size=1024,
        eval_batch_size=1024,
        test_batch_size=1024,
        lr=1e-3,
        weight_decay=1e-4,
        dataloader_drop_last=False,
        patience=1,
    )


@pytest.fixture
def generate_stock_dataset(generate_trainarguments):
    args = generate_trainarguments
    train_dates = (
        pl.datetime_range(
            datetime.datetime.strptime(args.train_date_range[0], "%Y-%m-%d"),
            datetime.datetime.strptime(args.train_date_range[1], "%Y-%m-%d"),
            datetime.timedelta(days=1),
            eager=True,
            time_unit="ms",
        )
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    eval_dates = (
        pl.datetime_range(
            datetime.datetime.strptime(args.eval_date_range[0], "%Y-%m-%d"),
            datetime.datetime.strptime(args.eval_date_range[1], "%Y-%m-%d"),
            datetime.timedelta(days=1),
            eager=True,
            time_unit="ms",
        )
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    test_dates = (
        pl.datetime_range(
            datetime.datetime.strptime(args.test_date_range[0], "%Y-%m-%d"),
            datetime.datetime.strptime(args.test_date_range[1], "%Y-%m-%d"),
            datetime.timedelta(days=1),
            eager=True,
            time_unit="ms",
        )
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    symbols = np.array([f"symbol{i:03d}" for i in range(50)])
    d, s = zip(*itertools.product(train_dates, symbols))
    d = np.array(d)
    s = np.array(s)
    x_seq = len(args.x_slots)
    y_seq = len(args.y_slots)
    x = np.random.randn(len(train_dates) * len(symbols), x_seq, len(args.x_columns))
    y = np.random.randn(len(train_dates) * len(symbols), y_seq, len(args.y_columns))
    train_set = StockDataset(
        date=d,
        symbol=s,
        x=x,
        y=y,
        y_columns=args.y_columns,
    )

    d, s = zip(*itertools.product(eval_dates, symbols))
    d = np.array(d)
    s = np.array(s)
    x_seq = len(args.x_slots)
    y_seq = len(args.y_slots)
    x = np.random.randn(len(eval_dates) * len(symbols), x_seq, len(args.x_columns))
    y = np.random.randn(len(eval_dates) * len(symbols), y_seq, len(args.y_columns))
    eval_set = StockDataset(
        date=d,
        symbol=s,
        x=x,
        y=y,
        y_columns=args.y_columns,
    )

    d, s = zip(*itertools.product(test_dates, symbols))
    d = np.array(d)
    s = np.array(s)
    x_seq = len(args.x_slots)
    y_seq = len(args.y_slots)
    x = np.random.randn(len(test_dates) * len(symbols), x_seq, len(args.x_columns))
    y = np.random.randn(len(test_dates) * len(symbols), y_seq, len(args.y_columns))
    test_set = StockDataset(
        date=d,
        symbol=s,
        x=x,
        y=y,
        y_columns=args.y_columns,
    )
    return train_set, eval_set, test_set


@pytest.fixture
def setup(generate_trainarguments, generate_stock_dataset):
    args = generate_trainarguments
    train_set, eval_set, test_set = generate_stock_dataset
    yield args, train_set, eval_set, test_set
    shutil.rmtree(args.save_dir)
    shutil.rmtree(args.dataset_dir)
