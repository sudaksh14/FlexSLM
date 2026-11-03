#!/usr/bin/bash
#SBATCH --job-name=eval_domain_ppl
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%x-%j.out

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"

python3 eval_domain_ppl.py \
  --configs "flexllama,refinedweb.kd_lambda05_1p4B;flexllama,gneissweb.kd_lambda05_1p4B;flexllama,pmc.kd_lambda05_3B" \
  --domains "pmc,refinedweb,gneissweb" \
  --batch_size 8
