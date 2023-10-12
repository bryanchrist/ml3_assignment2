#!/bin/bash
#SBATCH -A sds-phd-2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1

#SBATCH -t 0-08:00:00
#SBATCH -J ml3_assignment2_resnext_train
#SBATCH -o resnext_training-%A.out
#SBATCH -e resnext_training-%A.err

module purge

module load anaconda
module load cuda/11.4.2
module load singularity pytorch/2.0.1
source activate falcon_40B
pip install -U -r requirements.txt
pip install torchvision
conda install -y cudatoolkit
python densenet_test.py 