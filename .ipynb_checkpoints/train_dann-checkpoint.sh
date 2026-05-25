#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=downscaling
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=myjob.%j.log


source /projects/sds-lab/Shuochen/miniconda3/bin/activate
conda activate ai
python -u /home/wang.shuoc/downscaling/train.py