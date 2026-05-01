#!/usr/bin/bash
export HF_DATASETS_CACHE=/scratch-shared/$USER/hf_cache
export HF_HOME=/scratch-shared/$USER/hf_home
srun python3 $HOME/run_experiment.py run $1
