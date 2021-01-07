
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config '/home/hieu/lib/image_compression/configs/psnr_8192.yaml' \
    # --resume

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config '/home/hieu/lib/image_compression/configs/debug5.yaml' \
    --load '/mnt/HDD/runs__/debug5/checkpoints/iter_00131072_31.0966.pth' \
    --valid \
    # --save_output \