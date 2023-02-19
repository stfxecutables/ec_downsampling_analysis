#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=00-08:00:00
#SBATCH --job-name=tune
#SBATCH --output=tune_%j__%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

PROJECT=$SCRATCH/ec_downsampling_analysis
CODE=$PROJECT/src/tune.py

source "$PROJECT/.venv/bin/activate"
python "$CODE"