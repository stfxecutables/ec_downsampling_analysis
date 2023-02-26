#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=00-02:00:00
#SBATCH --job-name=mimic-iv
#SBATCH --output=uti_reduced_%j__%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --array=0-3
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

PROJECT=$SCRATCH/ec_downsampling_analysis
CODE=$PROJECT/scripts/downsample_slow_uti_reduced_array.py

source "$PROJECT/.venv/bin/activate"
python "$CODE"
