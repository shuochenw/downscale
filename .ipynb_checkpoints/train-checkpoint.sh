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

set -euo pipefail

source /projects/sds-lab/Shuochen/miniconda3/bin/activate
conda activate ai
python -u /home/wang.shuoc/downscaling/train.py \
    --rcm_var tmean \
    --gcm_name CanESM2 \
    --rcm_name CanRCM4 \
    --grid NAM-44i \
    --input_source gcm \
    --model_name SRResNet \
    --alpha_coarse 0.5 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --pin_memory \
    --amp \
    --log_every 5 \
    --study_name "srresnet_tmean_CanESM2_CanRCM4_NAM44i_gcm"
