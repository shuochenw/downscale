#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=rcan_shallow_aux
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=rcan_shallow_aux.%j.log

set -euo pipefail

source /projects/sds-lab/Shuochen/miniconda3/bin/activate
conda activate ai

RCM_VAR=tas
GCM_NAME=CanESM2
RCM_NAME=RCA4
GRID=NAM-44i
RCM_PRODUCT=raw
EXP=RCM_RCM
FACTOR=4

TRAIN_START_YEAR=1951
TRAIN_END_YEAR=2005
VAL_START_YEAR=2006
VAL_END_YEAR=2099
BATCH_SIZE=256
GRID_TAG=${GRID//-/}
RUN_NAME="rcan_hr_aux_shallowfusion_${RCM_VAR}_${GCM_NAME}_${RCM_NAME}_${GRID_TAG}_${RCM_PRODUCT}_${EXP}_bs${BATCH_SIZE}"
DATA_ROOT=/projects/sds-lab/Shuochen/downscaling/preprocessed

python -u /home/wang.shuoc/downscaling/train.py \
    --rcm_var "$RCM_VAR" \
    --gcm_name "$GCM_NAME" \
    --rcm_name "$RCM_NAME" \
    --grid "$GRID" \
    --rcm_product "$RCM_PRODUCT" \
    --exp "$EXP" \
    --factor "$FACTOR" \
    --data_root "$DATA_ROOT" \
    --input_file low_res.pth \
    --target_file high_res.pth \
    --hr_mask_file high_res_mask.pth \
    --hr_elevation_file high_res_elevation.pth \
    --train_start_year "$TRAIN_START_YEAR" \
    --train_end_year "$TRAIN_END_YEAR" \
    --val_start_year "$VAL_START_YEAR" \
    --val_end_year "$VAL_END_YEAR" \
    --n_trials 3000 \
    --epochs 300 \
    --early_stopping_patience 30 \
    --batch_size_choices "$BATCH_SIZE" \
    --num_features_choices 64 96 128 \
    --num_resblk_min 4 \
    --num_resblk_max 16 \
    --num_resblk_step 2 \
    --num_groups_choices 1 2 4 8 \
    --reduction_choices 4 8 16 \
    --res_scale_choices 0.05 0.1 0.2 \
    --learning_rate_min 1e-4 \
    --learning_rate_max 3e-4 \
    --weight_decay_min 1e-7 \
    --weight_decay_max 1e-3 \
    --num_workers 0 \
    --pin_memory \
    --amp \
    --grad_clip_norm 1.0 \
    --log_every 1 \
    --study_name "$RUN_NAME"
