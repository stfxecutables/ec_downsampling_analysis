from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
from pathlib import Path

from src.enumerables import ClassifierKind, Dataset
from src.prediction import evaluate_downsampling

if __name__ == "__main__":
    idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if idx is None:
        raise EnvironmentError("No SLURM_ARRAY_TASK_ID")

    kind = [*ClassifierKind][int(idx)]
    evaluate_downsampling(
        classifier=kind,
        dataset=Dataset.MimicIVReduced,
        downsample=True,
        n_reps=200,
        n_runs=10,
    )
