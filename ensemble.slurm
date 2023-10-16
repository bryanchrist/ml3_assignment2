#!/bin/bash
#SBATCH -A sds-phd-2022
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1

#SBATCH -t 0-05:00:00
#SBATCH -J ensemble
#SBATCH -o ensemble-%A.out
#SBATCH -e ensemble-%A.err

module purge

module load anaconda
module load cuda/11.4.2
module load singularity pytorch/2.0.1
source activate falcon_40B
pip install -U -r requirements.txt
pip install torchvision
conda install -y cudatoolkit
python ensemble.py 