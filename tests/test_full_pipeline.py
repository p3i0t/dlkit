import numpy as np

from dlkit.inference import InferenceArguments, InferencePipeline
from dlkit.train import StockTrainer

from tests.conftest import setup  # noqa: F401


def run_trainer(setup):  # noqa: F811
    """run training loop of trainer with setup fixture

    Args:
        setup pytest.fixture: setup elements

    Returns:
        bool: True if training is successful
    """
    args, train_set, eval_set, _ = setup
    trainer = StockTrainer(args=args, train_dataset=train_set, eval_dataset=eval_set)
    trainer.train()
    return True


def run_infer(setup):  # noqa: F811
    train_args, _, _, infer_set = setup
    infer_args = InferenceArguments(
        prod=train_args.prod,
        save_dir=train_args.save_dir,
        dataset_dir=train_args.dataset_dir,
        universe=train_args.universe,
        x_columns=train_args.x_columns,
        x_begin=train_args.x_begin,
        x_end=train_args.x_end,
        freq_in_min=train_args.freq_in_min,
        y_columns=train_args.y_columns,
        y_slots=train_args.y_slots,
        model=train_args.model,
        n_latest=3,
        device=train_args.device,
    )
    infer_pipe = InferencePipeline(args=infer_args)
    import polars as pl

    df_test = pl.DataFrame({"date": infer_set.date, "symbol": infer_set.symbol})
    # print(f"{infer_set.x.shape=}")
    n_features = len(train_args.x_flatten_columns)
    df_x = pl.DataFrame(
        np.reshape(infer_set.x, (-1, n_features)), schema=train_args.x_flatten_columns
    )
    # print(f"{df_x.shape=}")
    df_test = df_test.hstack(df_x)
    df_test = df_test.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d"))
    o = infer_pipe(df_test)
    print(o)
    return True


def test_train_infer(setup):  # noqa: F811
    assert run_trainer(setup) is True
    assert run_infer(setup) is True
