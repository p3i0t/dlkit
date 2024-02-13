from dlkit.inference import InferenceArguments


def test_initialize_infer_args():
    from pathlib import Path

    SAVE_DIR = Path("tests/save")
    DATA_DIR = Path("tests/data")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args = InferenceArguments(
        prod="test",
        universe="euniv_largemid",
        save_dir=SAVE_DIR,
        dataset_dir=DATA_DIR,
        x_columns=["open", "high", "low", "close", "volume"],
        x_begin="0930",
        x_end="1030",
        freq_in_min=10,
        y_columns=["ret"],
        y_slots="1030",
        model="GPT_small",
        date='today',
        n_latest=3,
        device="cuda",
    )
    assert args.x_shape == (6, 5)

    import shutil

    shutil.rmtree(SAVE_DIR)
    shutil.rmtree(DATA_DIR)
