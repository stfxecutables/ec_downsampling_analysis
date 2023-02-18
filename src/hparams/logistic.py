from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from typing import Any, Collection, Dict, List, Optional, Sequence, Union

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

SGD_TUNED: Dict[Dataset, Optional[Dict[str, Any]]] = {
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
    alpha: Optional[float] = 1e-4,
    l1_ratio: Optional[float] = 0.15,
    lr_init: Optional[float] = 1e-3,
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    average: bool = False,
) -> List[Hparam]:
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
        FixedHparam("n_jobs", value=1, default=1),
    ]


class SGDLRHparams(Hparams):
    def __init__(
        self,
        hparams: Union[Collection[Hparam], Sequence[Hparam], None] = None,
    ) -> None:
        if hparams is None:
            hparams = sgd_lr_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dataset: Dataset) -> Dict[str, Any]:
        hps = SGD_TUNED[dataset]
        if hps is None:
            return self.defaults().to_dict()
        return hps

    def set_n_jobs(self, n_jobs: int) -> None:
        self.hparams["n_jobs"].value = n_jobs
