import bisect
import datetime
import itertools
import os
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import polars as pl
import torch
from loguru import logger
from pydantic import Field

from dlkit.models import get_model
from dlkit.preprocessing import DataFrameNormalizer
from dlkit.exceptions import ModelDirectoryEmptyError

from dlkit.utils import CHECHPOINT_META
from dlkit.train import DatasetMetaArgs

logger = logger.bind(where="inference")

__all__ = ["InferenceArguments", "InferencePipeline"]


class InferenceArguments(DatasetMetaArgs):
    prod: str = Field(..., description="product (or experiment) name")
    save_dir: Path = Field(...)
    dataset_dir: Path = Field(...)
    universe: str = Field(...)
    model: str = Field(..., description="model name")
    n_latest: int = Field(3, description="number of latest models to ensemble")
    device: str = Field("cuda", description="Device to run on.")


class InferencePipeline:
    """The general inference pipeline for both offline and online inference."""

    def __init__(self, args: InferenceArguments):
        self.args = args
        import os
        model_dir = f"{args.save_dir}/{args.prod}"
        self.models_available = sorted(os.listdir(model_dir))
        if len(self.models_available) == 0:
            raise ModelDirectoryEmptyError(model_dir)

    def _load_ckpt(
        self, model_dir: str
    ) -> Tuple[torch.nn.Module, DataFrameNormalizer | None]:
        """Load model and normalizer from checkpoint directory.

        Args:
            model_dir (str): model directory.

        Returns:
            Tuple[torch.nn.Module, DataFrameNormalizer]: model and normalizer.
        """
        model = get_model(
            name=self.args.model, d_in=self.args.d_in, d_out=self.args.d_out
        ).to(self.args.device)
        ckpt_dir = f"{model_dir}/{CHECHPOINT_META.prefix_dir}"
        model.load_state_dict(torch.load(f"{ckpt_dir}/{CHECHPOINT_META.model}"))
        normalizer = None
        if os.path.isfile(f"{ckpt_dir}/{CHECHPOINT_META.normalizer}"):
            normalizer = torch.load(f"{ckpt_dir}/{CHECHPOINT_META.normalizer}")
        return model, normalizer

    def __call__(
        self, df: pl.DataFrame
    ) -> Dict[int, pl.DataFrame] | pl.DataFrame | None:
        df_dict = df.partition_by("date", as_dict=True)  # dict of [date, df_cs]

        dict_list = []
        for date in sorted(df_dict.keys()):
            df_cs = df_dict[date]
            df_pred = self._forward_cross_sectional_batch(date, df_cs)
            dict_list.append(df_pred)

        merged_dict = defaultdict(list)
        for e in dict_list:
            for n, _df in e.items():
                merged_dict[n].append(_df)
        o = {n: pl.concat(df_list) for n, df_list in merged_dict.items()}
        if len(o) > 1:
            return o
        return list(o.values())[0]

    def _forward_cross_sectional_batch(
        self, infer_date: datetime.datetime = None, df_cs: pl.DataFrame = None
    ) -> Dict[int, pl.DataFrame] | None:
        """
        Inference on a cross-sectional batch of data,
        which is not supposed to be called directly.

        Args:
            infer_date: The date of the cross-sectional batch.
            df_cs: The cross-sectional batch of dataframe.

        Returns:
            prediction: The prediction dataframe.
        """
        if df_cs is None:
            raise ValueError("cross-sectional df must be provided.")
        if infer_date is None:
            raise ValueError("date must be provided.")
        # df_cs = self.preprocess(df_cs)  # this is shared by all models
        df_index = df_cs.select(["date", "symbol"])

        # if isinstance(infer_date, datetime.datetime):
        _infer_date: str = infer_date.strftime("%Y-%m-%d")

        i = bisect.bisect_right(self.models_available, _infer_date)
        if i == 0:
            logger.error(f"no model available for date {infer_date}.")
            return None
        n_retain = (
            self.args.n_latest
            if isinstance(self.args.n_latest, int)
            else max(self.args.n_latest)
        )
        models_required = self.models_available[i - n_retain : i]
        logger.info(f"models required: {models_required}")

        pred_list = []
        for model_date in models_required:
            model_dir = f"{self.args.save_dir}/{self.args.prod}/{model_date}"
            _model, _normalizer = self._load_ckpt(model_dir)
            if _normalizer is not None:
                _df = _normalizer.transform(deepcopy(df_cs))
            else:
                _df = df_cs
            # _df = _normalizer.transform(deepcopy(df_cs))  # this is model specific
            x = (
                _df.select(self.args.x_slot_columns)
                .to_numpy(order="c")
                .reshape(-1, *self.args.x_shape)
            )
            x = torch.Tensor(x).to(self.args.device)
            pred = _model(x)[:, self.args.output_indices, :]
            pred = pred.detach().cpu().numpy()
            pred_list.append(pred)

        n_latest = (
            [self.args.n_latest]
            if isinstance(self.args.n_latest, int)
            else self.args.n_latest
        )
        res_dict = {}
        for n in n_latest:
            pred = np.stack(pred_list[-n:]).mean(axis=0)
            pred_flatten = pred.reshape((pred.shape[0], -1))
            df = df_index.hstack(pl.from_numpy(pred_flatten, schema=self.args.y_slot_columns))
            res_dict[n] = df

        return res_dict
