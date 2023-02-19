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
    MAX_WORKERS = {
        Dataset.MimicIV: {
            ClassifierKind.GBT: 30,
            ClassifierKind.SVM: 8,
            ClassifierKind.RF: 30,
            ClassifierKind.LR: 8,
        },
    }
    grid = [
        Namespace(**args)
        for args in list(
            ParameterGrid(
                {
                    "dataset": [Dataset.MimicIV],
                    "kind": [*ClassifierKind],
                    "only_rep": list(range(200)),
                }
            )
        )
    ]
    print(f"Total number of combinations: {len(grid)}")  # 800
    idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx is None:
        for args in grid:
            evaluate_downsampling(
                classifier=args.kind,
                dataset=args.dataset,
                downsample=True,
                n_reps=200,
                n_runs=10,
                max_workers=8,  # on local machine
                only_rep=args.only_rep,
            )

    else:
        args = grid[int(idx)]

        evaluate_downsampling(
            classifier=args.kind,
            dataset=args.dataset,
            downsample=True,
            n_reps=200,
            n_runs=10,
            max_workers=MAX_WORKERS[args.dataset][args.kind],
            only_rep=args.only_rep,
        )
