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
from src.prediction import evaluate_baselines

if __name__ == "__main__":
    datasets = [
        # Dataset.Diabetes,
        # Dataset.Transfusion,
        # Dataset.Parkinsons,
        # Dataset.SPECT,
        # Dataset.Diabetes130,
        # Dataset.Diabetes130Reduced,
        # Dataset.HeartFailure,
        # Dataset.MimicIV,
        Dataset.MimicIVReduced,
        # Dataset.UTIResistance,
        # Dataset.UTIResistanceReduced,
    ]
    MAX_WORKERS = 80 if os.environ.get("CC_CLUSTER") == "niagara" else 8
    grid = [
        Namespace(**args)
        for args in list(
            # ParameterGrid({"dataset": Dataset.fast(), "kind": [*ClassifierKind]})
            ParameterGrid({"dataset": datasets, "kind": [*ClassifierKind]})
            # ParameterGrid({"dataset": datasets, "kind": [ClassifierKind.GBT]})
        )
    ]
    print(f"Total number of combinations: {len(grid)}")
    idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx is None:
        for args in grid:
            evaluate_baselines(
                classifier=args.kind,
                dataset=args.dataset,
                n_reps=1,
                n_runs=500,
                max_workers=MAX_WORKERS,
            )

    else:
        args = grid[int(idx)]

        evaluate_baselines(
            classifier=args.kind,
            dataset=args.dataset,
            n_reps=500,
            n_runs=1,
            max_workers=MAX_WORKERS,
        )
