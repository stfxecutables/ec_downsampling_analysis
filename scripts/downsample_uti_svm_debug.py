from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
from argparse import Namespace
from pathlib import Path

from sklearn.model_selection import ParameterGrid

from src.enumerables import ClassifierKind, Dataset
from src.prediction import evaluate_downsampling

if __name__ == "__main__":
    MAX_WORKERS = 8
    evaluate_downsampling(
        classifier=ClassifierKind.SVM,
        dataset=Dataset.UTIResistance,
        downsample=True,
        n_reps=200,
        n_runs=10,
        max_workers=MAX_WORKERS,
        only_rep=None,
    )
