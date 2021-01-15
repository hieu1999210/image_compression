
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config '/home/hieu/lib/image_compression/configs/psnr_8192.yaml' \
    # --resume

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config '/home/hieu/lib/image_compression/configs/ssim_512.yaml' \
    --load '/mnt/HDD/runs__/ssim_512/checkpoints/iter_00129792_24.2206.pth' \
    --valid \
    --save_output \


# /mnt/HDD/runs__/psnr_8192/checkpoints/iter_00109824_37.6322.pth
# /mnt/HDD/runs__/psnr_256/checkpoints/iter_00130816_28.8680.pth