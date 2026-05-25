#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=diffusion
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=myjob.%j.log


source /projects/sds-lab/Shuochen/miniconda3/bin/activate
conda activate ai
python -u diffusion.py \
    --lr_path /projects/sds-lab/Shuochen/downscaling/CORDEX/tmean.CanESM2.CanRCM4.day.NAM-44i.mbcn-gridMET/coarse_4x.pth \
    --hr_path /projects/sds-lab/Shuochen/downscaling/CORDEX/tmean.CanESM2.CanRCM4.day.NAM-44i.mbcn-gridMET/high_res.pth \
    --mask_path /projects/sds-lab/Shuochen/downscaling/CORDEX/tmean.CanESM2.CanRCM4.day.NAM-44i.mbcn-gridMET/high_res_mask.pth \
    --save_dir ./ckpt_diffusion_tmean \
    --epochs 300 \
    --batch_size 16 \
    --timesteps 1000 \
    --base_channels 64
