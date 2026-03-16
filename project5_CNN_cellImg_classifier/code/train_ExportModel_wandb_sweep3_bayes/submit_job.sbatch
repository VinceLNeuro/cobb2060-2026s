#!/bin/bash
#SBATCH --job-name=moa_cnn_classifier_run6_ExportModel_sweep
#SBATCH --output=./out/log/moa_cnn_classifier_run6_ExportModel_sweep_%j.out
#SBATCH --error=./out/log/moa_cnn_classifier_run6_ExportModel_sweep_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
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

start_time=$(date +%s)


#### wandb login ####

#### Run [6] ####

# Register sweep on the wandb's server (on comp node)
# --------------------
# wandb sweep --project moa-cnn-classifier sweep_config_v3_bayes.yaml
# --------------------
# wandb: Creating sweep from: sweep_config_v3_bayes.yaml
# wandb: Creating sweep with ID: 6axw7aap
# wandb: View sweep at: https://wandb.ai/tvluo-university-of-pittsburgh/moa-cnn-classifier/sweeps/6axw7aap
# wandb: Run sweep agent with: wandb agent tvluo-university-of-pittsburgh/moa-cnn-classifier/6axw7aap

wandb agent --count 1 \
    tvluo-university-of-pittsburgh/moa-cnn-classifier/6axw7aap


end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total duration: $duration seconds"

