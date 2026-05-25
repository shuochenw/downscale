#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=vit
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=myjob.%j.log


source /projects/sds-lab/Shuochen/miniconda3/bin/activate
conda activate ai
python -u /home/wang.shuoc/downscaling/vit.py \
    --rcm_var tmean \
    --gcm_name CanESM2 \
    --rcm_name CanRCM4 \
    --grid NAM-44i \
    --factor 4 \
    --input_source gcm \
    --batch_size 128 \
    --n_trials 1000 \
    --epochs 300
