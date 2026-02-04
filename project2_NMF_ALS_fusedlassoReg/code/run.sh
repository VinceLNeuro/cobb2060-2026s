#!/usr/bin/env bash

######## Slurm resource allocation ########
#SBATCH --job-name=findBestHyperparam_NMF
#SBATCH --cluster=htc
#SBATCH --time=5:00:00
#SBATCH --nodes=1 #default - all cores on one machine
#SBATCH --ntasks-per-node=1 #default
#SBATCH --cpus-per-task=32 # number of cores (max)
##SBATCH --mem=500G         # 768G

##SBATCH --clusters=gpu  
##SBATCH --partition=a100    #a100,a100_nvlink,l40s
##SBATCH --constraint=a100,80g,amd
##SBATCH --gres=gpu:1      # number of GPUs per node (gres=gpu:N)  

#SBATCH --account=hpark
#SBATCH --mail-user=til177@pitt.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --output=./%x_slurm%A.out        #./output/%x-slurm_%A.out

######## Load software into environment ########
module purge
source ~/conda_init.sh
conda activate hugen_general

set -ev
# Confirm Python version and conda environment
echo "Using Python: $(which python)"
python -V
echo "Conda env: $CONDA_DEFAULT_ENV"

start_time=$(date +%s)

# Grid search
python nmf_als_fusedlasso.py -b ../data/chr10 -c chr10 -s 1000000 -e 1500000  -k 15
python nmf_als_fusedlasso.py -b ../data/chr10 -c chr10 -s 1000000 -e 11000000 -k 10

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Duration: $duration seconds"

