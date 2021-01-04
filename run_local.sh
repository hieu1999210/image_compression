
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config '/home/hieu/lib/image_compression/configs/psnr_8192.yaml' \
    # --resume

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config '/home/hieu/lib/image_compression/configs/debug4.yaml' \
#     --load '/mnt/HDD/runs__/debug4/checkpoints/iter_00130816_28.8680.pth' \
#     --valid \
#     --save_output \
