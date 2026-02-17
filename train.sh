# NOTE: change output path so we do not checkpoint on network drive!

# TODO: must use the same seed like 42 for later training

# python tools/train.py \
#     -f exps/example/mot/yolox_dancetrack_val.py \
#     -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth \
#     -o output_dir /ssd0/shenzhen/Methods/OC_SORT/YOLOX_outputs