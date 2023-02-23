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
    MAX_WORKERS = 80 if os.environ.get("CC_CLUSTER") == "niagara" else 8
    grid = [
        Namespace(**args)
        for args in list(
            # ParameterGrid({"dataset": [Dataset.HeartFailure], "kind": [*ClassifierKind]})
            ParameterGrid({"dataset": Dataset.very_fast(), "kind": [*ClassifierKind]})
            # ParameterGrid({"dataset": [Dataset.Diabetes], "kind": [*ClassifierKind]})
            # ParameterGrid({"dataset": [Dataset.Diabetes], "kind": [ClassifierKind.SVM]})
        )
    ]
    print(f"Total number of combinations: {len(grid)}")
    idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx is None:
        for args in grid:
            if args.dataset in [Dataset.Diabetes, Dataset.HeartFailure]:
                continue
            evaluate_downsampling(
                classifier=args.kind,
                dataset=args.dataset,
                downsample=True,
                n_reps=200,
                n_runs=10,
            )

    else:
        args = grid[int(idx)]

        evaluate_downsampling(
            classifier=args.kind,
            dataset=args.dataset,
            downsample=True,
            n_reps=200,
            n_runs=10,
            max_workers=MAX_WORKERS,
        )
