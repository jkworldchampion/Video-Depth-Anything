#!/usr/bin/env bash
set -euo pipefail

# # 1) diff=false, conv=false, conv_out_channel=0
# echo "===== Experiment 1: diff=false, conv=false ====="
# python -m train_experiment --conv_out_channel 0

# echo "===== Experiment 2: diff=true, conv=false ====="
# # 2) diff=true,  conv=false, conv_out_channel=0
# python -m train_experiment  --diff --conv_out_channel 0

# # 3) diff=true,  conv=true,  conv_out_channel=32
# echo "===== Experiment 3: diff=true, conv=true, conv_out_channel=32 ====="
# python -m train_experiment  --diff --conv --conv_out_channel 32

# # 4) diff=true,  conv=true,  conv_out_channel=64
# echo "===== Experiment 3: diff=true, conv=true, conv_out_channel=64 ====="
# python -m train_experiment  --diff --conv --conv_out_channel 64



# 1) diff=false, conv=false, conv_out_channel=0
echo "===== Experiment 1: diff=false, conv=false ====="
#python -m train_experiment --conv_out_channel 0 --pretrained_ckpt ""

#python -m train_experiment --conv_out_channel 0 --pretrained_ckpt ""

# 2) diff=true,  conv=false, conv_out_channel=0
echo "===== Experiment 2: diff=true, conv=false ====="
#python -m train_experiment_juhwan  --diff --conv_out_channel 0 --pretrained_ckpt "video_depth_anything_vits.pth"

#python -m train_experiment_juhwan  --diff --conv_out_channel 0 --pretrained_ckpt "video_depth_anything_vits.pth"


# 3) diff=true,  conv=true,  conv_out_channel=32
echo "===== Experiment 3: diff=true, conv=true, conv_out_channel=32 ====="
python -m train_experiment  --diff --conv_out_channel 32 --pretrained_ckpt "video_depth_anything_vits.pth"

#python -m train_experiment  --diff --conv_out_channel 32 --pretrained_ckpt ""

#python -m train_experiment  --diff --conv_out_channel 32 --pretrained_ckpt ""

