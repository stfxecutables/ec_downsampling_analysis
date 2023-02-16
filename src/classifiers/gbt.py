from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.random import Generator
from pandas import DataFrame, Series
from xgboost import XGBClassifier

from src.classifiers.classifier import Classifier
from src.hparams.gbt import XGBoostHparams


class XGBoostClassifier(Classifier):
    def __init__(
        self,
        hparams: XGBoostHparams,
    ) -> None:
        self.hparams = hparams
        self.classifier = XGBClassifier(**self.hparams.to_dict())

    def fit(self, X: DataFrame, y: Series, rng: Optional[Generator]) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.classifier.fit(X, y)

    def score(self, X: DataFrame, y: Series) -> float:
        return float(self.classifier.score(X, y))
