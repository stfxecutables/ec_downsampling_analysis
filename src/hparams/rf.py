from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from typing import Any, Collection, Dict, List, Optional, Sequence, Union

from src.enumerables import Dataset
from src.hparams.hparams import (
    ContinuousHparam,
    FixedHparam,
    Hparam,
    Hparams,
    OrdinalHparam,
)

TUNED: Dict[Dataset, Optional[Dict[str, Any]]] = {
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
    eta: Optional[float] = None,
    lamda: Optional[float] = None,
    alpha: Optional[float] = None,
    num_round: Optional[int] = None,
    gamma: Optional[float] = None,
    colsample_bylevel: Optional[float] = None,
    colsample_bynode: Optional[float] = None,
    colsample_bytree: Optional[float] = None,
    max_depth: Optional[int] = None,
    max_delta_step: Optional[int] = None,
    min_child_weight: Optional[float] = None,
    subsample: Optional[float] = None,
) -> List[Hparam]:
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
        FixedHparam("n_jobs", value=1),
    ]


class XGBRFHparams(Hparams):
    def __init__(
        self,
        hparams: Union[Collection[Hparam], Sequence[Hparam], None] = None,
    ) -> None:
        if hparams is None:
            hparams = xgboost_rf_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dataset: Dataset) -> Dict[str, Any]:
        hps = TUNED[dataset]
        if hps is None:
            return self.defaults().to_dict()
        return hps

    def set_n_jobs(self, n_jobs: int) -> None:
        self.hparams["n_jobs"].value = n_jobs
