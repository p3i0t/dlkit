import torch

from dlkit.models import get_model, list_all_models


def test_get_registered_models():
    models = list_all_models()

    assert set(models) == set(
        ["GPT_small", "GPT_medium", "GPT_large", "LSTM_small", "LSTM_medium"]
    )


def test_get_model():
    m = get_model("GPT_small", d_in=5, d_out=1)
    assert isinstance(m, torch.nn.Module)
    m = get_model("LSTM_small", d_in=5, d_out=1)
    assert isinstance(m, torch.nn.Module)
    m = get_model("GPT_medium", d_in=5, d_out=1)
    assert isinstance(m, torch.nn.Module)
    m = get_model("GPT_large", d_in=5, d_out=1)
    assert isinstance(m, torch.nn.Module)
