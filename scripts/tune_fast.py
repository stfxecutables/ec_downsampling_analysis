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
from src.tune import random_tune

if __name__ == "__main__":
    grid = [
        Namespace(**args)
        for args in list(
            ParameterGrid(
                {
                    "dataset": [Dataset.Diabetes],
                    # "dataset": Dataset.very_fast(),
                    "kind": [
                        # ClassifierKind.GBT,
                        # ClassifierKind.LR,
                        ClassifierKind.SVM,
                        # ClassifierKind.RF,
                    ],
                }
            )
        )
    ]
    print(f"Total number of combinations: {len(grid)}")
    idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx is None:
        # raise ValueError("Not in array job. $SLURM_ARRAY_TASK_ID is undefined.")
        for args in grid:
            random_tune(
                classifier=args.kind,
                dataset=args.dataset,
                metric=Metric.Accuracy,
                n_runs=1000,
                force=True,
            )

    else:
        args = grid[int(idx)]
        random_tune(
            classifier=args.kind,
            dataset=args.dataset,
            metric=Metric.Accuracy,
            n_runs=250,
            force=False,
        )
