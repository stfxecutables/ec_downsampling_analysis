#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=00-00:45:00
#SBATCH --job-name=val_slow
#SBATCH --output=eval_slow_%j__%A_%a.out
#SBATCH --array=0-199
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

PROJECT=$SCRATCH/ec_downsampling_analysis
CODE=$PROJECT/scripts/downsample_slow_uti.py

source "$PROJECT/.venv/bin/activate"
python "$CODE"