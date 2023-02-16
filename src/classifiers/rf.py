from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

from pandas import DataFrame, Series
from xgboost import XGBRFClassifier

from src.classifiers.classifier import Classifier
from src.hparams.rf import XGBRFHparams


class XGBoostRFClassifier(Classifier):
    def __init__(
        self,
        hparams: XGBRFHparams,
    ) -> None:
        self.hparams = hparams
        self.classifier = XGBRFClassifier(**self.hparams.to_dict())

    def fit(self, X: DataFrame, y: Series) -> None:
        self.classifier.fit(X, y)

    def score(self, X: DataFrame, y: Series) -> float:
        return float(self.classifier.score(X, y))
