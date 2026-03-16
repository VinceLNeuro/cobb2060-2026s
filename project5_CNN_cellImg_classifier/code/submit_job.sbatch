#!/bin/bash
#SBATCH --job-name=moa_cnn_classifier_run3ExportModel
#SBATCH --output=./out_run3_default_epoch30_imgSize128_ExportModel/moa_cnn_classifier_run3ExportModel_%j.out
#SBATCH --error=./out_run3_default_epoch30_imgSize128_ExportModel/moa_cnn_classifier_run3ExportModel_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=til177@pitt.edu


## start an ipcluster instance and launch jupyter server    
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

#### RUN 1 ####
# python train.py --train_data_dir /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project5_CNN_cellImg_classifier/data/train \
#                 --out /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project5_CNN_cellImg_classifier/code/model \
#                 --max_epochs 1 \
#                 --img_size 512 \

# #### RUN 2 - see if ExportModel class can help close the gap between val and test ####
# python train_ExportModel.py \
#     --train_data_dir /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project5_CNN_cellImg_classifier/data/train \
#     --out /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project5_CNN_cellImg_classifier/code/out_run2_default_epoch1_imgSize512_ExportModel/model_ExportModel \
#     --max_epochs 1 \
#     --img_size 512


#### RUN 3 - RUN 2 is underfitting -> increase epochs and reduce img size (faster) ####
python train_ExportModel.py \
    --train_data_dir /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project5_CNN_cellImg_classifier/data/train \
    --out /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project5_CNN_cellImg_classifier/code/out_run3_default_epoch30_imgSize128_ExportModel/model_ExportModel \
    --max_epochs 30 \
    --img_size 128


end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total duration: $duration seconds"

