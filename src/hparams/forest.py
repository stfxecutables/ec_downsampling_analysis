from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Any, Collection, Sequence

from src.enumerables import Dataset
from src.hparams.hparams import (
    ContinuousHparam,
    FixedHparam,
    Hparam,
    Hparams,
    OrdinalHparam,
)

TUNED: dict[Dataset, dict[str, Any] | None] = {
    Dataset.Diabetes: None,
    Dataset.Diabetes130: None,
    Dataset.HeartFailure: None,
    Dataset.MimicIV: None,
    Dataset.Parkinsons: None,
    Dataset.SPECT: None,
    Dataset.Transfusion: None,
    Dataset.UTIResistance: None,
}


def xgboost_rf_hparams(
    eta: float | None = None,
    lamda: float | None = None,
    alpha: float | None = None,
    num_round: int | None = None,
    gamma: float | None = None,
    colsample_bylevel: float | None = None,
    colsample_bynode: float | None = None,
    colsample_bytree: float | None = None,
    max_depth: int | None = None,
    max_delta_step: int | None = None,
    min_child_weight: float | None = None,
    subsample: float | None = None,
) -> list[Hparam]:
    """Note we define ranges as per https://arxiv.org/abs/2106.11189 Appendix B.2,
    Table 6

    See https://xgboost.readthedocs.io/en/latest/parameter.html for defaults. When
    defaults conflict with the tuning ranges given above, we choose a default as
    close as possible to the XGBoost default.
    """
    return [
        ContinuousHparam(
            "eta",
            eta,
            max=1.0,
            min=0.001,
            log_scale=True,
            default=0.3,
        ),
        ContinuousHparam(
            "lambda",
            lamda,
            max=1.0,
            min=1e-10,
            log_scale=True,
            default=1.0,
        ),
        ContinuousHparam(
            "alpha",
            alpha,
            max=1.0,
            min=1e-10,
            log_scale=True,
            default=1e-10,
        ),
        # XGB complains below are unused
        # OrdinalHparam("num_round", num_round, max=1000, min=1),
        ContinuousHparam(
            "gamma",
            gamma,
            max=1.0,
            min=0.1,
            log_scale=True,
            default=0.1,
        ),
        ContinuousHparam(
            "colsample_bylevel",
            colsample_bylevel,
            max=1.0,
            min=0.1,
            log_scale=False,
            default=1.0,
        ),
        ContinuousHparam(
            "colsample_bynode",
            colsample_bynode,
            max=1.0,
            min=0.1,
            log_scale=False,
            default=1.0,
        ),
        ContinuousHparam(
            "colsample_bytree",
            colsample_bytree,
            max=1.0,
            min=0.1,
            log_scale=False,
            default=1.0,
        ),
        OrdinalHparam(
            "max_depth",
            max_depth,
            max=20,
            min=1,
            default=6,
        ),
        OrdinalHparam("max_delta_step", max_delta_step, max=10, min=0, default=0),
        ContinuousHparam(
            "min_child_weight",
            min_child_weight,
            max=20,
            min=0.1,
            log_scale=True,
            default=1,
        ),
        ContinuousHparam(
            "subsample",
            subsample,
            max=1.0,
            min=0.01,
            log_scale=False,
            default=1,
        ),
        FixedHparam(
            "enable_categorical",
            value=True,
        ),
        FixedHparam("tree_method", value="hist"),
    ]


class XGBRFHparams(Hparams):
    def __init__(
        self,
        hparams: Collection[Hparam] | Sequence[Hparam] | None = None,
    ) -> None:
        if hparams is None:
            hparams = xgboost_rf_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dataset: Dataset) -> dict[str, Any]:
        hps = TUNED[dataset]
        if hps is None:
            return self.defaults().to_dict()
        return hps
