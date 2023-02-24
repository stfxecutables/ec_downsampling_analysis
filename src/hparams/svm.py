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
    OrdinalHparam,
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


def svm_hparams(
    C: Optional[float] = 1.0,
    gamma: Optional[float] = 0.1,
    shrinking: Optional[bool] = False,
) -> List[Hparam]:
    # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
    # for a possible tuning range on C, gamma
    return [
        ContinuousHparam(
            "C",
            C,
            max=1e3,
            min=1e-5,
            log_scale=True,
            default=1.0,
        ),
        ContinuousHparam("gamma", gamma, max=1e3, min=1e-10, log_scale=True, default=0.1),
        CategoricalHparam(
            "shrinking",
            value=shrinking,
            categories=[True, False],
            default="False",
        ),
        FixedHparam("kernel", value="rbf", default="rbf"),
        FixedHparam("probability", value=False, default=False),
        # FixedHparam("gamma", value="auto", default="auto"),
    ]


class SVMHparams(Hparams):
    def __init__(
        self,
        hparams: Union[Collection[Hparam], Sequence[Hparam], None] = None,
    ) -> None:
        if hparams is None:
            hparams = svm_hparams()
        super().__init__(hparams)

    def tuned_dict(self, dataset: Dataset) -> Dict[str, Any]:
        hps = SGD_TUNED[dataset]
        if hps is None:
            return self.defaults().to_dict()
        return hps

    def set_n_jobs(self, n_jobs: int) -> None:
        pass
