[![DOI](https://zenodo.org/badge/414786742.svg)](https://zenodo.org/badge/latestdoi/414786742)

# Installation (Cross-Platform)

First install [Poetry](https://python-poetry.org/docs/#installation). Open a terminal, `cd` to
the cloned repository directory, and run

```sh
poetry install
```

to setup and install the virtual environment.

# Basic Testing


**WARNING**: Tests will use all your available cores and spew a huge amount of text
to your terminal, so only run them if you are prepared.

```sh
poetry run pytest
```

# Running the Paper Analyses

## Compute Canada

The code to run the analyses here is unfortunately deeply tied up with Compute Canada cluster
specifics and SLURM. Normally, once you have activated your virtual environment with e.g.
`poetry shell`, the procedure would be to run:

```sh
python analysis/create_jobscripts.py
sbatch analysis/job_scripts/submit_all_downsampling.sh
sbatch analysis/job_scripts/submit_all_feature.sh
sbatch analysis/job_scripts/submit_mlp_downsampling.sh
sbatch analysis/job_scripts/submit_mlp_feature.sh
```

There are hard-coded switches that modify resource requests and runtimes depending on the cluster,
so reproducing this would unfortunately require reading and modifying the code.

## Running Locally

Alternately, you can run a specific dataset analysis by specifying command-line arguments and
faking the environment. For example (assuming you have run `poetry shell`):

```sh
CC_CLUSTER=niagara \
python analysis/feature_downsampling.py \
  --classifier=lr \
  --dataset=diabetes \
  --kfold-reps=50 \
  --n-percents=200 \
  --results-dir=<your_directory_here> \
  --cpus=8 \
  --pbar
```

Don't expect this to work on Windows, and it is most likely unfeasible to run all analyses like this
on a single machine.

