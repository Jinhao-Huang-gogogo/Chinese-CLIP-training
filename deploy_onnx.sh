export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

mkdir -p /data/jinhaohuang/Chinese-CLIP/deploy/ # 创建ONNX模型的输出文件夹

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-B-16 \
       --pytorch-ckpt-path /data/jinhaohuang/Chinese-CLIP/pretrained_weights/clip_cn_vit-b-16.pt \
       --save-onnx-path /data/jinhaohuang/Chinese-CLIP/deploy/vit-b-16 \
       --convert-text --convert-vision

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-H-14 \
       --pytorch-ckpt-path /data/jinhaohuang/Chinese-CLIP/pretrained_weights/clip_cn_vit-h-14.pt \
       --save-onnx-path /data/jinhaohuang/Chinese-CLIP/deploy/vit-h-14 \
       --convert-text --convert-vision


python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-L-14 \
       --pytorch-ckpt-path /data/jinhaohuang/Chinese-CLIP/pretrained_weights/clip_cn_vit-l-14.pt \
       --save-onnx-path /data/jinhaohuang/Chinese-CLIP/deploy/vit-l-14 \
       --convert-text --convert-vision

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-L-14-336 \
       --pytorch-ckpt-path /data/jinhaohuang/Chinese-CLIP/pretrained_weights/clip_cn_vit-l-14-336.pt \
       --save-onnx-path /data/jinhaohuang/Chinese-CLIP/deploy/vit-l-14-336 \
       --convert-text --convert-vision

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch RN50 \
       --pytorch-ckpt-path /data/jinhaohuang/Chinese-CLIP/pretrained_weights/clip_cn_rn50.pt \
       --save-onnx-path /data/jinhaohuang/Chinese-CLIP/deploy/rn50 \
       --convert-text --convert-vision