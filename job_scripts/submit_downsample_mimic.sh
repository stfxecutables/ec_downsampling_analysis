#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=00-23:59:59
#SBATCH --job-name=mimic-iv
#SBATCH --output=mimic-iv_%j__%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

PROJECT=$SCRATCH/ec_downsampling_analysis
CODE=$PROJECT/scripts/downsample_mimic.py

source "$PROJECT/.venv/bin/activate"
python "$CODE"
