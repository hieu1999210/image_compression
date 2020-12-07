
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config '/home/hieu/lib/image_compression/configs/debug.yaml' \
    --resume
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config '/mnt/DATA/fixmatch/configs/test_cxr11.yaml' \
#     # --resume \
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config '/mnt/DATA/fixmatch/configs/test_cxr8.yaml' \

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config '/home/hieu/lib/custom_fcos/configs/debug5f_gn.yaml' \
#     --load '/mnt/HDD/coco_runs/debug5f_gn/checkpoints/epoch_014_26.0458.pth' \
#     --valid \
# cxr_fixmatch
