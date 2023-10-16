#!/bin/bash
#SBATCH -A ds5559
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1

#SBATCH -t 0-05:00:00
#SBATCH -J ml3_assignment2_resnext_train
#SBATCH -o resnext1600_training-%A.out
#SBATCH -e resnext1600_training-%A.err

module purge

module load anaconda
module load cuda/11.4.2
module load singularity pytorch/2.0.1
source activate falcon_40B
pip install -U -r requirements.txt
pip install torchvision
conda install -y cudatoolkit
python resnext1600.py 