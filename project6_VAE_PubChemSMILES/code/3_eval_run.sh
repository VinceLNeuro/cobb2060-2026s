#!/bin/bash
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=eval_test_%j.out

## load anaconda
module load anaconda
eval "$(conda shell.bash hook)"
conda activate csb_gen

set -ev
# Confirm Python version and conda environment
echo "Using Python: $(which python)"
python -V
echo "Conda env: $CONDA_DEFAULT_ENV"

# python 3_eval.py \
#     --model /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run0_noSweep/vae_decoder_gen.pth \
#     --evals 10

# python 3_eval.py \
#     --model /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run4_longMask_v3_genEval_default/test_on_small/vae_decoder_len50_Mol50k_test_smallerVAE_rm.pth \
#     --evals 10
#### takes forever to run ####


# python 3_eval.py \
#     --model /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run3_long_noSweep_Masking/vae_decoder_gen.pth \
#     --evals 10

# python 3_eval.py \
#     --model /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run4_longMask_v3_genEval_default/debug_v2_to_v3/vae_decoder_len50_Mol50k_smallerVAE.pth \
#     --evals 10


#### best model so far ####
python 3_eval.py \
    --model /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run4_longMask_v3_genEval_default/vae_decoder_len150_Mol5M_default.pth \
    --evals 1000
