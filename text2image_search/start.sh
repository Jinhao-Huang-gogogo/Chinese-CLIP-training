#!/bin/bash

# 启动CLIP文搜图服务脚本

# export PYTHONPATH=/data/jinhaohuang/Chinese-CLIP:$PYTHONPATH

python app.py \
    --model_path /data/jinhaohuang/Chinese-CLIP/experiments/car-V0.1-people_V1.1_finetune_vit-l-14_roberta-base-small-lr/checkpoints/epoch3.pt \
    --image_data /data/jinhaohuang/Chinese-CLIP/datasets/car-V0.1/vehicle_train.tsv \
    --host 0.0.0.0 \
    --port 5000 \
    --device cuda:0


    