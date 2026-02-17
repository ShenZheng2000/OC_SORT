# installations (for rest, follow the author)
# conda create -n ocsort python=3.8 -y
# conda activate ocsort
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip install "numpy<1.24"

# create dataset symlinks
# ln -s /ssd0/shenzhen/Datasets/tracking datasets

# TODO: use seed of 42 for inference (re-run twice to see if reproduce)
# NOTE: don't write _val in exp_name (messy path!)

# exp_name=yolox_dancetrack_pretrained
# python tools/run_ocsort_dance.py \
#     -f exps/example/mot/yolox_dancetrack_val.py \
#     -c pretrained/bytetrack_dance_model.pth.tar \
#     -b 1 -d 1 --fp16 --fuse --expn $exp_name \
#     --dataset dancetrack \
#     --seed 42

# exp_name=yolox_dancetrack_1
# CUDA_VISIBLE_DEVICES=1 python tools/run_ocsort_dance.py \
#     -f exps/example/mot/yolox_dancetrack_val.py \
#     -c YOLOX_outputs/yolox_dancetrack_1/best_ckpt.pth.tar \
#     -b 1 -d 1 --fp16 --fuse --expn $exp_name \
#     --dataset dancetrack \
#     --seed 42

# exp_name=yolox_dancetrack_2
# CUDA_VISIBLE_DEVICES=2 python tools/run_ocsort_dance.py \
#     -f exps/example/mot/yolox_dancetrack_val.py \
#     -c YOLOX_outputs/yolox_dancetrack_2/best_ckpt.pth.tar \
#     -b 1 -d 1 --fp16 --fuse --expn $exp_name \
#     --dataset dancetrack \
#     --seed 42