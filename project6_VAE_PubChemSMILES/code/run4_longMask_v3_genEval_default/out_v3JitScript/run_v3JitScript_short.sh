#!/bin/bash
#SBATCH --job-name=run_v3JitScript_short__defaultVAE
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g006
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=til177@pitt.edu


## load anaconda
module load anaconda
eval "$(conda shell.bash hook)"
conda activate csb_gen

set -ev
# Confirm Python version and conda environment
echo "Using Python: $(which python)"
python -V
echo "Conda env: $CONDA_DEFAULT_ENV"


# wandb login
# source .venv/bin/activate
start_time=$(date +%s)
# DATA_DIR="/net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/data"

# # Smaller set
# python 1_preprocess.py \
#     ${DATA_DIR}/pubchem_randomized.smi.gz \
#     ${DATA_DIR}/pubchem_randomized_len50_Mol50k.npy \
#     --max_len 50 --max_mols 50000

# # Full set, but with less molecules, to speed up training
# python 1_preprocess.py \
#     ${DATA_DIR}/pubchem_randomized.smi.gz \
#     ${DATA_DIR}/pubchem_randomized_len150_Mol5M.npy \
#     --max_len 150 --max_mols 5000000


####################    This run    ####################
# total=50k -> train=47500 -> step per epoch = 93 -> total steps = 930
#           -> val  =2500  -> 5 batches

# # Smaller Mod
# python train_VAE_v3RmNoGrad.py \
#         -T /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/data/pubchem_randomized_len50_Mol50k.npy \
#         --out /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run4_longMask_v3_genEval_default/out_v3RmNoGrad/vae_decoder_len50_Mol50k_smallerVAE_rmNoGrad.pth \
#         --max_length 50 \
#         --bidirectional \
#         --embedding_dim 64 --hidden_size 256 --num_layers 1 \
#         --epochs 10 --batch_size 512 \
#         --log_every 30 \
#         --val_every 93 \
#         --val_batches 5 \
#         --gen_eval_samples 100

# Larger Mod
python train_VAE_v3JitScript.py \
        -T /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/data/pubchem_randomized_len50_Mol50k.npy \
        --out /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run4_longMask_v3_genEval_default/out_v3JitScript/vae_decoder_len50_Mol50k_defaultVAE_JitScript.pth \
        --max_length 50 \
        --bidirectional \
        --epochs 10 --batch_size 512 \
        --log_every 30 \
        --val_every 93 \
        --val_batches 5 \
        --gen_eval_samples 100



end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total duration: $duration seconds"

