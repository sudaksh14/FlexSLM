#!/usr/bin/env bash
#SBATCH --job-name=flexslm_fakequant
#SBATCH --partition=fatq
#SBATCH --nodelist=node201
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# Expected (pessimistic) 30 min on CPU: 1.7B + 4 narrower slices + the 0.6B full width.
#SBATCH --time=01:00:00
#SBATCH --output=jobs/%x-%j.out
# Weight-only fake int8 of MedNest exports (eval/fake_quant.py), shared scales for the narrower levels.
source scripts/das6/common.sh cpu
python -m pytest test/test_fake_quant.py -q -p no:cacheprovider
S=hf_export/student__sft.medmix_qwen3_1p7b_10L_kv_teacher
O=hf_export/int8__student_1p7b
python -m eval.fake_quant $S/level9 $O/level9 --slice $S/level7 $O/level7 $S/level6 $O/level6 $S/level4 $O/level4 $S/level2 $O/level2
A=hf_export/flexllama__sft.paper.qwen3_0p6b_nested.no_stage1
python -m eval.fake_quant $A/level2 hf_export/int8__a2_0p6b/level2 --slice $A/level1 hf_export/int8__a2_0p6b/level1 $A/level0 hf_export/int8__a2_0p6b/level0
echo "[fakequant] done"
