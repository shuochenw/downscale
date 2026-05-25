#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=rcan_dann_2_aux
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=rcan_dann_2_aux.%j.log

set -euo pipefail

source /projects/sds-lab/Shuochen/miniconda3/bin/activate
conda activate ai

RCM_VAR=tas
GCM_NAME=CanESM2
RCM_NAME=RCA4
GRID=NAM-44i
RCM_PRODUCT=raw
FACTOR=4
EXP=GCM_RCM

TRAIN_START_YEAR=1951
TRAIN_END_YEAR=2005
VAL_START_YEAR=2071
VAL_END_YEAR=2099
BATCH_SIZE_CHOICES=(256 512)
LAMBDA_GRL_MAX=0.3

case "${VAL_START_YEAR}_${VAL_END_YEAR}" in
    2006_2040)
        ALPHA_DOMAIN_MIN=2e-5
        ALPHA_DOMAIN_MAX=5e-4
        DOMAIN_HIDDEN_DIM_CHOICES=(32 64)
        ;;
    2041_2070)
        ALPHA_DOMAIN_MIN=3e-4
        ALPHA_DOMAIN_MAX=1.5e-3
        DOMAIN_HIDDEN_DIM_CHOICES=(32 64)
        ;;
    2071_2099)
        ALPHA_DOMAIN_MIN=2e-4
        ALPHA_DOMAIN_MAX=8e-4
        DOMAIN_HIDDEN_DIM_CHOICES=(64 128)
        ;;
    *)
        ALPHA_DOMAIN_MIN=2e-5
        ALPHA_DOMAIN_MAX=1.5e-3
        DOMAIN_HIDDEN_DIM_CHOICES=(32 64 128)
        ;;
esac

DATA_ROOT=/projects/sds-lab/Shuochen/downscaling/preprocessed
EXP_DIR="$DATA_ROOT/${RCM_VAR}.${GCM_NAME}.${RCM_NAME}.day.${GRID}.${RCM_PRODUCT}.${EXP}"
MODEL_NAME=RCAN_HR_Aux_ShallowFusion_DANN
RUN_NAME="dann_${VAL_START_YEAR}_${VAL_END_YEAR}_2"
SAVE_DIR="$EXP_DIR/trained_models/$MODEL_NAME/$RUN_NAME"

python -u /home/wang.shuoc/downscaling/train_dann_2.py \
    --rcm_var "$RCM_VAR" \
    --gcm_name "$GCM_NAME" \
    --rcm_name "$RCM_NAME" \
    --grid "$GRID" \
    --rcm_product "$RCM_PRODUCT" \
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
    --early_stopping_patience 10 \
    --batch_size_choices "${BATCH_SIZE_CHOICES[@]}" \
    --num_features_choices 192 224 256 \
    --num_resblk_min 18 \
    --num_resblk_max 24 \
    --num_resblk_step 2 \
    --num_groups_choices 1 2 4 8 \
    --reduction_choices 16 32 \
    --res_scale_choices 0.05 0.1 0.2 0.3 0.4 \
    --learning_rate_min 2e-4 \
    --learning_rate_max 5e-4 \
    --weight_decay_min 1e-7 \
    --weight_decay_max 1e-3 \
    --alpha_domain_min "$ALPHA_DOMAIN_MIN" \
    --alpha_domain_max "$ALPHA_DOMAIN_MAX" \
    --lambda_grl_max "$LAMBDA_GRL_MAX" \
    --domain_hidden_dim_choices "${DOMAIN_HIDDEN_DIM_CHOICES[@]}" \
    --num_workers 0 \
    --pin_memory \
    --amp \
    --grad_clip_norm 1.0 \
    --log_every 1 \
    --study_name "$RUN_NAME" \
    --save_dir "$SAVE_DIR"
