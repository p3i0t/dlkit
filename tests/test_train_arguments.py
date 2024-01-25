from dlkit.train import TrainArguments


def test_initialize_train_args():
    from pathlib import Path

    SAVE_DIR = Path("tests/save")
    DATA_DIR = Path("tests/data")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainArguments(
        prod="test",
        universe="euniv_largemid",
        save_dir=SAVE_DIR,
        dataset_dir=DATA_DIR,
        milestone="2020-01-01",
        x_columns=["open", "high", "low", "close", "volume"],
        x_begin="0930",
        x_end="1030",
        freq_in_min=10,
        y_columns=["ret"],
        y_slots="1030",
        train_date_range=("2020-01-01", "2021-12-31"),
        eval_date_range=("2022-01-01", "2022-12-31"),
        test_date_range=None,
        normalizer="zscore",
        monitor_metric="loss",
        monitor_mode="min",
        model="GPT_small",
        epochs=20,
        seed=42,
        device="cuda",
        train_batch_size=1024,
        eval_batch_size=1024,
        test_batch_size=1024,
        lr=1e-3,
        weight_decay=1e-4,
        dataloader_drop_last=False,
        patience=6,
    )
    assert args.x_shape == (6, 5)
    assert args.d_in == 5
    assert args.d_out == 1
    import shutil

    shutil.rmtree(SAVE_DIR)
    shutil.rmtree(DATA_DIR)
