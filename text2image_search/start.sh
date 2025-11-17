#!/bin/bash

# 启动CLIP文搜图服务脚本

# export PYTHONPATH=/data/jinhaohuang/Chinese-CLIP:$PYTHONPATH

python app.py \
    --model_path /data/jinhaohuang/Chinese-CLIP/experiments/PA100K_finetune_vit-l-14_roberta-base/checkpoints/epoch7.pt \
    --image_data /data/jinhaohuang/Chinese-CLIP/text2image_search/recognition_from_video/results/all_detections_base64.tsv \
    --host 0.0.0.0 \
    --port 5000 \
    --device cuda:0


    