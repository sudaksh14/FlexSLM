#!/usr/bin/env bash
#SBATCH --job-name=flexslm_fetch
#SBATCH --partition=fatq
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
# Expected (pessimistic) 1 h: ~31 GB over HTTPS from SURFdrive; limit 2x that.
#SBATCH --time=02:00:00
#SBATCH --output=jobs/%x-%j.out
# Student's earlier code, results and four trained models (public SURFdrive share, user 2026-11-11).
source scripts/das6/common.sh cpu
OUT=/var/scratch/$USER/flexslm/old_ckpts
T=Fbc3MqrabSMEAKT
python scripts/fetch_surfdrive.py $T $OUT repo --exclude /.git/ /__pycache__/ repo/data/ repo/nanoGPT repo/nanochat
python scripts/fetch_surfdrive.py $T $OUT \
  saved_models/flexxxxllamaxnsftxmmedmixxxx_qwen3_1p7b_teacher.pt \
  saved_models/flexxxxllamaxnsftxmmedmixxxx_qwen3_1p7b_flexxxxslm.pt \
  saved_models/flexxxxllamaxnsftxmmedmixxxx_qwen3_1p7b_10L_teacher.pt \
  saved_models/flexxxxllamaxnsftxmmedmixxxx_qwen3_1p7b_10L_shard.pt
