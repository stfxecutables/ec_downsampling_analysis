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
from xgboost import XGBRFClassifier

from src.classifiers.classifier import Classifier
from src.enumerables import Metric
from src.hparams.rf import XGBRFHparams


class XGBoostRFClassifier(Classifier):
    def __init__(
        self,
        hparams: XGBRFHparams,
    ) -> None:
        self.hparams = hparams
        self.classifier = XGBRFClassifier(**self.hparams.to_dict())

    def fit(self, X: DataFrame, y: Series, rng: Optional[Generator]) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.classifier.fit(X.to_numpy(), y.to_numpy())

    # def score(self, X: DataFrame, y: Series, metric: Metric = Metric.Accuracy) -> float:
    #     return float(self.classifier.score(X, y))
