#!/usr/bin/bash
LOCAL_CACHE=/local_scratch/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL_CACHE"
trap 'rm -rf "$LOCAL_CACHE"' EXIT

export HF_DATASETS_CACHE=$LOCAL_CACHE/hf_cache
export HF_HOME=$LOCAL_CACHE/hf_home
srun python3 "$SLURM_SUBMIT_DIR/run_experiment.py" run $1
