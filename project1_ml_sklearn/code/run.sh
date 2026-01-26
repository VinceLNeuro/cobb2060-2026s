#!/usr/bin/env bash

######## Slurm resource allocation ########
#SBATCH --job-name=singleGeneMut_classifier
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

# python singleGeneMut_classifier_run.py -e brca_tcga/data_mrna_seq_v2_rsem.txt -m brca_tcga/data_mutations.txt -p brca_tcga/data_mrna_seq_v2_rsem.txt  -g RYR2  AKAP9 DNAH11 TP53 UTRN HERC2 DNAH2 PIK3CA -o tcga_tcga
python singleGeneMut_classifier_run2.py -e brca_tcga/data_mrna_seq_v2_rsem.txt -m brca_tcga/data_mutations.txt -p brca_tcga/data_mrna_seq_v2_rsem.txt  -g TP53 DNAH2 PIK3CA -o tcga_tcga

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Duration: $duration seconds"

