# -*- coding: utf-8 -*-
import os
import json
import base64
import torch
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import torch.nn.functional as F
import sys
import time

# 获取当前文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPService:
    def __init__(self, model_path, image_data_path, vision_model="ViT-L-14", 
                 text_model="RoBERTa-wwm-ext-base-chinese", context_length=52, device="cuda:0"):
        self.device = device
        self.context_length = context_length
        self.vision_model = vision_model
        self.text_model = text_model
        
        # 初始化模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 加载图片特征
        self.image_features, self.image_ids, self.image_base64_dict = self._load_image_features(image_data_path)
        
        logger.info(f"Loaded {len(self.image_ids)} image features")

    def _load_model(self, model_path):
        """加载CLIP模型"""
        # 导入模型配置
        from cn_clip.clip.model import convert_weights, CLIP
        from cn_clip.training.main import convert_models_to_fp32
        
        # 加载模型配置
        vision_model_config_file = Path(__file__).parent / f"../cn_clip/clip/model_configs/{self.vision_model.replace('/', '-')}.json"
        text_model_config_file = Path(__file__).parent / f"../cn_clip/clip/model_configs/{self.text_model.replace('/', '-')}.json"
        
        with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
            model_info = json.load(fv)
            if isinstance(model_info['vision_layers'], str):
                model_info['vision_layers'] = eval(model_info['vision_layers'])        
            for k, v in json.load(ft).items():
                model_info[k] = v

        # 创建模型
        model = CLIP(**model_info)
        convert_weights(model)
        convert_models_to_fp32(model)
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
        model.load_state_dict(sd)
        
        # 移动到设备
        model = model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
        return model

    def _load_image_features(self, image_data_path):
        """加载图片特征或从TSV文件提取"""
        feature_cache_path = image_data_path.replace('.tsv', '_features.npy')
        id_cache_path = image_data_path.replace('.tsv', '_ids.npy')
        base64_cache_path = image_data_path.replace('.tsv', '_base64_dict.npy')
        
        # 如果缓存存在，直接加载
        if os.path.exists(feature_cache_path) and os.path.exists(id_cache_path):
            logger.info("Loading cached image features...")
            image_features = np.load(feature_cache_path)
            image_ids = np.load(id_cache_path)
            image_base64_dict = np.load(base64_cache_path, allow_pickle=True).item()
            return torch.from_numpy(image_features), image_ids.tolist(), image_base64_dict
        
        # 否则从TSV文件提取
        logger.info("Extracting image features from TSV file...")
        return self._extract_features_from_tsv(image_data_path, feature_cache_path, id_cache_path, base64_cache_path)

    def _extract_features_from_tsv(self, tsv_path, feature_cache_path, id_cache_path, base64_cache_path):
        """从TSV文件提取图片特征"""
        image_features_list = []
        image_ids_list = []
        image_base64_dict = {}
        
        batch_size = 64
        current_batch = []
        current_batch_ids = []
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            num = 0
            for line in tqdm(f, desc="Processing images"):
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                    
                img_id = parts[0]
                img_base64 = parts[1]
                
                try:
                    # 解码base64图像
                    img_data = base64.b64decode(img_base64)
                    image = Image.open(io.BytesIO(img_data)).convert('RGB')
                    
                    # 预处理图像
                    processed_image = self._preprocess_image(image)
                    
                    current_batch.append(processed_image)
                    current_batch_ids.append(img_id)
                    image_base64_dict[img_id] = img_base64
                    
                    # 批量处理
                    if len(current_batch) >= batch_size:
                        batch_tensor = torch.stack(current_batch).to(self.device)
                        with torch.no_grad():
                            batch_features = self.model(batch_tensor, None)
                            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                        
                        image_features_list.append(batch_features.cpu())
                        image_ids_list.extend(current_batch_ids)
                        
                        current_batch = []
                        current_batch_ids = []
                        
                except Exception as e:
                    logger.warning(f"Error processing image {img_id}: {e}")
                    continue
                num += 1
                if num > 5000:
                    break
        
        # 处理最后一批
        if current_batch:
            batch_tensor = torch.stack(current_batch).to(self.device)
            with torch.no_grad():
                batch_features = self.model(batch_tensor, None)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            
            image_features_list.append(batch_features.cpu())
            image_ids_list.extend(current_batch_ids)
        
        # 合并所有特征
        image_features = torch.cat(image_features_list, dim=0)
        
        # 保存缓存
        np.save(feature_cache_path, image_features.numpy())
        np.save(id_cache_path, np.array(image_ids_list))
        np.save(base64_cache_path, image_base64_dict)
        
        logger.info(f"Saved features cache to {feature_cache_path}")
        
        return image_features, image_ids_list, image_base64_dict

    def _preprocess_image(self, image):
        """预处理图像"""
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
        
        return preprocess(image)

    def encode_text(self, text):
        """编码文本"""
        from cn_clip.clip import tokenize
        
        with torch.no_grad():
            text_tokens = tokenize([text], context_length=self.context_length).to(self.device)
            text_features = self.model(None, text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu()

    def search_images(self, text, topk=10, confidence_threshold=0.0):
        """搜索最相似的图片"""
        # 编码文本
        text_features = self.encode_text(text)
        
        # 计算相似度
        similarities = (text_features @ self.image_features.T).squeeze(0)
        
        # 应用置信度阈值过滤
        if confidence_threshold > 0:
            mask = similarities >= confidence_threshold
            filtered_indices = torch.nonzero(mask, as_tuple=True)[0]
            filtered_similarities = similarities[filtered_indices]
            
            # 在过滤后的结果中取topk
            if len(filtered_similarities) > 0:
                topk_filtered = min(topk, len(filtered_similarities))
                topk_values, topk_relative_indices = torch.topk(filtered_similarities, topk_filtered)
                topk_indices = filtered_indices[topk_relative_indices]
            else:
                topk_values = torch.tensor([])
                topk_indices = torch.tensor([])
        else:
            # 不使用阈值，直接取topk
            topk_values, topk_indices = torch.topk(similarities, min(topk, len(self.image_ids)))
        
        results = []
        for score, idx in zip(topk_values.tolist(), topk_indices.tolist()):
            img_id = self.image_ids[idx]
            results.append({
                'image_id': img_id,
                'score': score,
                'base64': self.image_base64_dict[img_id]
            })
        
        return results

    def save_image_from_base64(self, base64_str, filepath):
        """将base64图片保存为文件"""
        try:
            img_data = base64.b64decode(base64_str)
            with open(filepath, 'wb') as f:
                f.write(img_data)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False

# 进度条工具
class tqdm:
    def __init__(self, iterable=None, desc=None):
        self.iterable = iterable
        self.desc = desc
        self.count = 0
        
    def __iter__(self):
        if self.desc:
            print(self.desc)
        for item in self.iterable:
            self.count += 1
            if self.count % 100 == 0:
                print(f"Processed {self.count} items")
            yield item

# 其他测试函数保持不变...