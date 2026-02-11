#!/usr/bin/env bash

######## Slurm resource allocation ########
#SBATCH --job-name=sc_clustering_predCellOntology_pcaBased
#SBATCH --cluster=htc
#SBATCH --time=1:00:00
#SBATCH --nodes=1 #default - all cores on one machine
#SBATCH --ntasks-per-node=1 #default
#SBATCH --cpus-per-task=12 # number of cores (max)
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

# set up data directory
DATADIR='/ihome/hpark/til177/GitHub/cobb2060-2026s/Data_cobb2060/proj3'


######## RUN & TIME ########
start_time=$(date +%s)

echo "=== Task 1: Unsupervised clustering (k=4) ==="
t1=$(date +%s)
python sc_clustering_predCellOntology_pcaBased.py -d $DATADIR/mouse_spatial_brain_section1_modified.h5ad -k 4 -o clusters_1.npy # unsupervised clustering
echo "Task 1 duration: $(($(date +%s) - t1)) seconds"


echo "=== Task 2: Unsupervised clustering (k=8) ==="
t2=$(date +%s)
python sc_clustering_predCellOntology_pcaBased.py -d $DATADIR/mouse_cortex_methods_comparison_log1p_cpm_modified.h5ad -k 8 -o clusters_2.npy
echo "Task 2 duration: $(($(date +%s) - t2)) seconds"


echo "=== Task 3: Supervised labeling (brain section) ==="
t3=$(date +%s)
python sc_clustering_predCellOntology_pcaBased.py -t $DATADIR/mouse_spatial_brain_section0.h5ad -d $DATADIR/mouse_spatial_brain_section1_modified.h5ad -o clusters_3.npy
echo "Task 3 duration: $(($(date +%s) - t3)) seconds"


echo "=== Task 4: Supervised labeling (cross-domain) ==="
t4=$(date +%s)
python sc_clustering_predCellOntology_pcaBased.py -t $DATADIR/mouse_spatial_brain_section0.h5ad -d $DATADIR/mouse_cortex_methods_comparison_log1p_cpm_modified.h5ad -o clusters_4.npy
echo "Task 4 duration: $(($(date +%s) - t4)) seconds"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total duration: $duration seconds"

