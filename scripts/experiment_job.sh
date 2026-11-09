#!/usr/bin/bash
LOCAL_CACHE=/scratch-local/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL_CACHE"
trap 'rm -rf "$LOCAL_CACHE"' EXIT

export HF_DATASETS_CACHE=$LOCAL_CACHE/hf_cache
export HF_HOME=$LOCAL_CACHE/hf_home
srun python3 -m training.run_experiment run $1 $2
