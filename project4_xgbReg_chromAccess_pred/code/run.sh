#!/bin/bash

######## Slurm resource allocation ########
#SBATCH --job-name=xgbReg_chromAccess_pred
#SBATCH -p dept_cpu
#SBATCH -t 1:00:00
#SBATCH --nodes=1 #default - all cores on one machine
#SBATCH --ntasks-per-node=1 #default
#SBATCH --cpus-per-task=44 # number of cores (max)
##SBATCH --mem=256G         # 768G

##SBATCH --clusters=gpu  
##SBATCH --partition=a100    #a100,a100_nvlink,l40s
##SBATCH --constraint=a100,80g,amd
##SBATCH --gres=gpu:1      # number of GPUs per node (gres=gpu:N)  

#SBATCH --output=./%x_slurm%A.out        #./output/%x-slurm_%A.out

######## Load software into environment ########
module load anaconda
eval "$(conda shell.bash hook)"
## activate conda env
conda activate csb_gen

set -ev
# Confirm Python version and conda environment
echo "Using Python: $(which python)"
python -V
echo "Conda env: $CONDA_DEFAULT_ENV"

# set up data directory
DATADIR='/net/dali/home/mscbio/til177/Github/cobb2060-2026s/project4_/data'


######## RUN & TIME ########
start_time=$(date +%s)

python assign4.py -g $DATADIR/mm10.fa -t $DATADIR/train.small.bed -m $DATADIR/cisBP_mouse.homer -p $DATADIR/test.small.bed -o out_small.npy
python assign4.py -g $DATADIR/mm10.fa -t $DATADIR/train.B.bed -m $DATADIR/cisBP_mouse.homer -p $DATADIR/test.B.bed -o out_b.npy

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total duration: $duration seconds"

