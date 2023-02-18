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

from src.enumerables import ClassifierKind, Dataset, Metric
from src.prediction import evaluate_downsampling

if __name__ == "__main__":
    MAX_WORKERS = {
        Dataset.UTIResistance: {
            ClassifierKind.GBT: 30,
            ClassifierKind.SVM: 8,
            ClassifierKind.RF: 30,
            ClassifierKind.LR: 8,
        }
    }
    grid = [
        Namespace(**args)
        for args in list(
            ParameterGrid({"dataset": [Dataset.fast()], "kind": [*ClassifierKind]})
        )
    ]
    print(f"Total number of combinations: {len(grid)}")
    idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx is None:
        raise ValueError("Not in array job. $SLURM_ARRAY_TASK_ID is undefined.")
    args = grid[int(idx)]

    evaluate_downsampling(
        classifier=args.kind,
        dataset=args.dataset,
        downsample=True,
        n_reps=200,
        n_runs=10,
        max_workers=8,
    )
