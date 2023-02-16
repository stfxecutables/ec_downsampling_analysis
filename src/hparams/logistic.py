from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import (
    Any,
    Collection,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src.constants import SKLEARN_SGD_LR_DEFAULT as LR_DEFAULT
from src.constants import SKLEARN_SGD_LR_MAX as LR_MAX
from src.constants import SKLEARN_SGD_LR_MIN as LR_MIN
from src.enumerables import Dataset
from src.hparams.hparams import (
    CategoricalHparam,
    ContinuousHparam,
    FixedHparam,
    Hparam,
    Hparams,
)

SGD_TUNED: dict[Dataset, dict[str, Any] | None] = {
    Dataset.Diabetes: None,
    Dataset.Diabetes130: None,
    Dataset.HeartFailure: None,
    Dataset.MimicIV: None,
    Dataset.Parkinsons: None,
    Dataset.SPECT: None,
    Dataset.Transfusion: None,
    Dataset.UTIResistance: None,
}


def sgd_lr_hparams(
    alpha: float | None = 1e-4,
    l1_ratio: float | None = 0.15,
    lr_init: float | None = 1e-3,
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    average: bool = False,
) -> list[Hparam]:
    # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
    # for a possible tuning range on C, gamma
    return [
        ContinuousHparam(
            "alpha", alpha, max=1e-1, min=1e-7, log_scale=True, default=1e-4
        ),
        ContinuousHparam(
            "l1_ratio", l1_ratio, max=1.0, min=0.0, log_scale=False, default=0.15
        ),
        ContinuousHparam(
            "eta0", lr_init, max=LR_MAX, min=LR_MIN, log_scale=True, default=LR_DEFAULT
        ),
        CategoricalHparam(
            "penalty",
            value=penalty,
            categories=["l1", "l2", "elasticnet", None],
            default="l2",
        ),
        CategoricalHparam("average", average, categories=[True, False], default=False),
        FixedHparam("loss", value="log_loss", default="log_loss"),
        FixedHparam("learning_rate", value="adaptive", default="adaptive"),
    ]


class SGDLRHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:
        if hparams is None:
            hparams = sgd_lr_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dataset: Dataset) -> dict[str, Any]:
        hps = SGD_TUNED[dataset]
        if hps is None:
            return self.defaults().to_dict()
        return hps
