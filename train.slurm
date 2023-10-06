#!/bin/bash
#SBATCH -A sds-phd-2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1

#SBATCH -t 3-00:00:00
#SBATCH -J ml3_assignment2_train
#SBATCH -o training-%A.out
#SBATCH -e training-%A.err

module purge

module load anaconda
module load cuda/11.4.2
module load singularity pytorch/2.0.1
source activate falcon_40B
conda install -y cudatoolkit
python train.py 