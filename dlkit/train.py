import copy
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Literal

import numpy as np
import polars as pl
import torch
from loguru import logger
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
)
from torch.optim import AdamW
from typing_extensions import Annotated

from dlkit.models import get_model
from dlkit.data.data_utils import (
    EvalPrediction,
    NumpyStockDataLoader,
    StockBatch,
    StockDataset,
)
from dlkit.utils import (
    CHECHPOINT_META,
    get_time_slots,
)
from dlkit.checks import check_epochs, check_milestone, check_normalizer

logger = logger.bind(where="trainer")


__all__ = ["DatasetMetaArgs", "TrainArguments", "StockTrainer"]


class DatasetMetaArgs(BaseModel):
    x_columns: list[str] = Field(..., description="input columns.")
    x_begin: str = Field(..., description="start time slot.")
    x_end: str = Field(..., description="end time slot.")
    freq_in_min: Literal[1, 10] = Field(
        10, description="frequency of input in minutes."
    )
    y_columns: list[str] = Field(..., description="output columns.")
    y_slots: str | list[str] = Field(..., description="output slots.")
    
    @property
    def x_slots(self) -> list[str]:
        return get_time_slots(
            start=self.x_begin, end=self.x_end, freq_in_min=self.freq_in_min
        )

    @property
    def d_in(self) -> int:
        return len(self.x_columns)

    @property
    def d_out(self) -> int:
        return len(self.y_columns)

    @property
    def output_indices(self) -> list[int]:
        import bisect

        if isinstance(self.y_slots, str):
            y_slots = [self.y_slots]
        elif isinstance(self.y_slots, list):
            y_slots = self.y_slots
        else:
            raise ValueError(f"y_slots {self.y_slots} is not a valid type.")
        o = [bisect.bisect(self.x_slots, _y) - 1 for _y in y_slots]
        assert all(e >= 0 for e in o)
        return o

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return len(self.x_slots), len(self.x_columns)

    @property
    def x_slot_columns(self) -> list[str]:
        return [
            f"{_f}_{_s}" for _s, _f in itertools.product(self.x_slots, self.x_columns)
        ]

    @property
    def y_shape(self) -> Tuple[int, ...]:
        if isinstance(self.y_slots, str):
            y_slots = [self.y_slots]
        elif isinstance(self.y_slots, list):
            y_slots = self.y_slots
        else:
            raise ValueError(f"y_slots {self.y_slots} is not a valid type.")
        return len(y_slots), len(self.y_columns)
    
    @property
    def y_slot_columns(self) -> list[str]:
        if isinstance(self.y_slots, str):
            y_slots = [self.y_slots]
        elif isinstance(self.y_slots, list):
            y_slots = self.y_slots
        else:
            raise ValueError(f"y_slots {self.y_slots} is not a valid type.")
        return [
            f"{_f}_{_s}" for _s, _f in itertools.product(y_slots, self.y_columns)
        ]
    

class TrainArguments(DatasetMetaArgs):
    """The class including all the arguments related to training.

    Args:
        prod (str): product (or experiment) name
        dataset_dir (Path): dataset directory
        universe (str): universe to train.
        milestone (Annotated[str, AfterValidator(check_date)]): milestone date of the model
        train_date_range (Tuple[str, str]): training date range.
        eval_date_range (Optional[Tuple[str, str]]): evaluation date range.
        test_date_range (Optional[Tuple[str, str]]): test date range.
        normalizer (Annotated[str, AfterValidator(check_normalizer)]): normalizer name.
        model (str): model name.
        epochs (Annotated[int, AfterValidator(check_epochs)]): training epochs.
        seed (int): universal seed for this training.
        lr (PositiveFloat): learning rate.
        weight_decay (PositiveFloat): weight decay.
        train_batch_size (PositiveInt): training batch size.
        eval_batch_size (PositiveInt): evaluation batch size.
        test_batch_size (PositiveInt): test batch size.
        dataloader_drop_last (bool): drop last batch in dataloader.
        device (str): device to run on.
        monitor_metric (str): metric to monitor.
        monitor_mode (str): mode to monitor.
        patience (int): number of epochs to wait before early stop.
        save_dir (Path): save directory for checkpoint, logs, etc.
        d_in (Annotated[int, "number of input features of the model, equals to the len of x_columns"]): input dimension.
        d_out (Annotated[int, "output dim of the model"]): output dimension.
        x_slots (list[str]): input slots.
        x_shape (Tuple[int, ...]): input shape.
        x_features (list[str]): input features.
        milestone_dir (Path): directory of this milestone for checkpoint and logs.
        output_indices (list[int]): output indices.
    """
    prod: str = Field(..., description="product (or experiment) name.")
    dataset_dir: Path = Field(..., description="dataset directory")
    universe: str = Field(..., description="universe to train.")
    milestone: Annotated[str, AfterValidator(check_milestone)] = Field(
        "2020-01-01", description="milestone date of the model"
    )
    train_date_range: Tuple[str, str] = Field(...)
    eval_date_range: Tuple[str, str] = Field(...)
    test_date_range: Optional[Tuple[str, str]] = Field(None)
    normalizer: Annotated[str, AfterValidator(check_normalizer)] = Field(
        "zscore", description="normalizer name."
    )

    # model training arguments
    model: str = Field(..., description="model name")
    epochs: Annotated[int, AfterValidator(check_epochs)] = Field(
        20, description="training epochs."
    )
    seed: int = Field(42, description="universal seed for this training.")
    lr: PositiveFloat = Field(5.0e-5)
    weight_decay: PositiveFloat = Field(1.0e-3)
    train_batch_size: PositiveInt = Field(1024)
    eval_batch_size: PositiveInt = Field(2048)
    test_batch_size: PositiveInt = Field(2048)
    dataloader_drop_last: bool = Field(False)
    device: str = Field("cuda", description="Device to run on.")
    monitor_metric: str = Field("loss", description="Metric to monitor.")
    monitor_mode: str = Field("min", description="Mode to monitor.")
    patience: int = Field(6, description="Number of epochs to wait before early stop.")
    save_dir: Path = Field(..., description="save directory for checkpoint, logs, etc.")

    @field_validator("universe")
    def validate_universe(cls, v: str):
        univ_list = [
            "euniv_largemid",
            "euniv_research",
        ]
        assert v in univ_list, f"{v} is not a valid universe"
        return v

    # @field_validator("save_dir")
    # def validate_save_dir(cls, v: Path):
    #     if v.exists():
    #         v.rmdir()
    #     v.mkdir(parents=True, exist_ok=False)
    #     return v

    @field_validator("dataset_dir")
    def validate_dataset_dir(cls, v: Path):
        assert v.exists(), f"{v} does not exist."
        return v
        
    @property
    def milestone_dir(self) -> Path:
        """directory of this milestone for checkpoint and logs"""
        return self.save_dir.joinpath(self.prod).joinpath(self.milestone)


class StockTrainer:
    """
    This is a specialized Trainer class for stock forcasting.

    ...
    Attributes
    ----------
    model: nn.Module
            The model to be trained.
    args: TrainArguments
            The arguments to be used for training.
    train_dataset: StockDataset
            The training dataset.
    eval_dataset: StockDataset
            The evaluation dataset.

    Methods
    -------
    get_eval_dataloader(eval_dataset: StockDataset = None) -> Iterable[StockBatch]
            Get the evaluation dataloader.
    get_train_dataloader() -> Iterable[StockBatch]
            Get the training dataloader.
    create_optimizer()
            Create the optimizer.
    train_epoch()
            Train the model for one epoch.
    eval_epoch(dataloader: Iterable[StockBatch]) -> dict[str, Any]
            Evaluate the model for one epoch.
    _process_batch(batch: StockBatch) -> StockBatch
            Process the batch.
    train()
            Train the model.
    save_checkpoint(save_dir: str)
            Save the checkpoint.
    """

    def __init__(
        self,
        *,
        args: TrainArguments,
        train_dataset: StockDataset = None,
        eval_dataset: StockDataset = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):
        self.args = args
        self.model = get_model(
            name=args.model, d_in=args.d_in, d_out=args.d_out
            ).to(self.args.device)
        # model.train()
        if self.model is None:
            raise ValueError("Trainer: model cannot be None.")
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=1e-3
        )
        self.compute_metrics = compute_metrics  # TODO

    def get_eval_dataloader(
        self, eval_dataset: Optional[StockDataset] = None
    ) -> Iterable[StockBatch]:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return NumpyStockDataLoader(
            dataset=eval_dataset,
            shuffle=False,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            # device=self.args.device,
        )

    def get_train_dataloader(self) -> Iterable[StockBatch]:
        """
        Get the training dataloader, i.e. transforming the training dataset into batches.

        Returns:
                Iterator[StockBatch]: The training dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires an eval dataset.")
        return NumpyStockDataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            # device=self.args.device,
        )

    def train_epoch(self):
        self.model.train()
        loss_list = []
        loader = self.get_train_dataloader()
        logger.info("loader prepared.")
        for batch_idx, batch in enumerate(loader):
            print(f"batch_idx: {batch_idx}")
            print(batch)
            x, y = self._process_xy(batch.x, batch.y)
            pred: torch.Tensor = self.model(x)[:, self.args.output_indices, :]
            loss = (pred - y).pow(2).mean()
            loss_list.append(loss.detach().cpu().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss = np.mean(loss_list)
        return loss

    def eval_epoch(
        self, dataloader: Iterable[StockBatch], prediction_loss_only: bool = False
    ) -> dict[str, Any]:
        self.model.eval()

        res_dict = defaultdict(list)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x, y = self._process_xy(batch.x, batch.y)
                pred: torch.Tensor = self.model(x)[:, self.args.output_indices, :]
                loss = (pred - y).pow(2).mean()
                res_dict["loss"].append(loss.detach().cpu().item())

                if not prediction_loss_only: # todo: modify when y is not single slot
                    res_dict["date"].append(batch.date)
                    res_dict["symbol"].append(batch.symbol)
                    pred_cols = [f"pred_{c}" for c in batch.y_columns]
                    df_pred = pl.DataFrame(
                        pred.detach().cpu().flatten(start_dim=1).numpy(),
                        schema=pred_cols,
                    )
                    res_dict["pred"].append(df_pred)
                    df_y = pl.DataFrame(
                        y.detach().cpu().flatten(start_dim=1).numpy(),
                        schema=batch.y_columns,
                    )
                    res_dict["y"].append(df_y)

            output = {"loss": np.mean(res_dict["loss"])}
            if not prediction_loss_only:
                df_ds = pl.DataFrame(
                    {
                        "date": np.concatenate(res_dict["date"]),
                        "symbol": np.concatenate(res_dict["symbol"]),
                    }
                )

                df_pred = pl.concat(res_dict["pred"])
                df_y = pl.concat(res_dict["y"])
                df = df_ds.hstack(df_pred).hstack(df_y)
                ic = df.group_by("date").agg(
                    pl.corr(a, b) for a, b in zip(df_y.columns, df_pred.columns)
                )
                ic = ic.select(df_y.columns).mean()

                ic_dict = {y: ic.get_column(y).item() for y in df_y.columns}
                del df, df_pred, df_y, df_ds, ic
            else:
                ic_dict = {}

            return output | ic_dict

    def _process_xy(self, x, y=None):
        """
        Process the batch, i.e. move the batch to the device and convert the batch to tensor.
        Args:
                batch (StockBatch): The stock batch to be processed.

        Returns:
                StockBatch: The processed batch.
        """
        x = torch.Tensor(x).to(self.args.device, non_blocking=True)
        x = torch.nan_to_num(x, nan=0.0)
        if y is not None:
            y = torch.Tensor(y).to(self.args.device, non_blocking=True)
            y = torch.nan_to_num(y, nan=0.0)
            return x, y
        else:
            return x

    def train(self):
        args = self.args
        if args.monitor_mode == "min":
            best = float("inf")
        else:
            best = float("-inf")
        best_epoch = 0
        best_state = None

        for epoch in range(int(self.args.epochs)):
            train_loss = self.train_epoch()
            eval_dict = self.eval_epoch(self.get_eval_dataloader(), prediction_loss_only=True)
            logger.info(f"======> Epoch {epoch:02d}")
            logger.info(f"train_loss: {train_loss:.4f}")
            logger.info("Evaluation:")
            for k, v in eval_dict.items():
                logger.info(f"eval, {k}: {v:.4f}")

            if (
                args.monitor_mode == "max" and eval_dict[args.monitor_metric] > best
            ) or (args.monitor_mode == "min" and eval_dict[args.monitor_metric] < best):
                logger.info("======> New best")
                best = eval_dict[args.monitor_metric]
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(str(args.milestone_dir))
            # logger.info(f"Saved checkpoint to {args.save_dir}.")

            if epoch - best_epoch >= args.patience:
                logger.info("======> Early stop.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _save_checkpoint(self, save_dir: str) -> None:
        """
        Save the checkpoint, including model state, training arguments.
        Args:
            save_dir: output directory of this trainer.
        Returns:

        """
        checkpoint_dir = f"{save_dir}/{CHECHPOINT_META.prefix_dir}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"{checkpoint_dir}/{CHECHPOINT_META.model}",
        )
        torch.save(self.args, f"{checkpoint_dir}/{CHECHPOINT_META.training_args}")

    def evaluate(self, eval_dataset: Optional[StockDataset] = None) -> dict[str, Any]:
        """
        Evaluate the model on the given dataset.

        Parameters
        ----------
        eval_dataset: StockDataset
                The evaluation dataset.

        Returns
        -------
        dict[str, Any]
                The evaluation results.
        """
        loader = self.get_eval_dataloader(eval_dataset)
        return self.eval_epoch(loader)


if __name__ == "__main__":
    ...
