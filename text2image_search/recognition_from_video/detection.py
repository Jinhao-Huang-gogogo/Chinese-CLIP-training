from ultralytics import YOLO
import cv2
import os
import numpy as np
from typing import List, Tuple, Optional
import logging
import base64

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonDetector:
    """
    使用YOLOv8进行人物检测
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        初始化YOLO模型

        Args:
            model_name: YOLO模型名称或路径
        """
        self.model = YOLO(model_name)
        logger.info(f"加载YOLO模型: {model_name}")

    def detect_persons(self, image_path: str, conf_threshold: float = 0.5) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        检测图像中的人物

        Args:
            image_path: 图像文件路径
            conf_threshold: 置信度阈值

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
                检测到的人物图像和边界框列表 (image, (x1, y1, x2, y2))
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {image_path}")

        # 使用YOLO进行检测
        results = self.model(image, verbose=False)

        detected_persons = []

        for result in results:
            for box in result.boxes:
                # 只检测'person'类别 (class_id=0)
                if int(box.cls) == 0 and box.conf >= conf_threshold:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 裁剪人物区域
                    person_img = image[y1:y2, x1:x2]

                    detected_persons.append((person_img, (x1, y1, x2, y2)))

                    logger.info(f"检测到人物: 置信度={box.conf.item():.2f}, 位置=({x1}, {y1}, {x2}, {y2})")

        return detected_persons

    def detect_persons_in_video_frames(self, frame_paths: List[str], conf_threshold: float = 0.5) -> List[List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        """
        批量检测视频帧中的人物

        Args:
            frame_paths: 视频帧路径列表
            conf_threshold: 置信度阈值

        Returns:
            List[List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
                每帧检测到的人物图像和边界框列表
        """
        all_detections = []

        for frame_path in frame_paths:
            detections = self.detect_persons(frame_path, conf_threshold)
            all_detections.append(detections)

        return all_detections

    def save_detected_persons(self, detections: List[Tuple[np.ndarray, Tuple[int, int, int, int]]], output_dir: str, base_name: str = "person") -> List[str]:
        """
        保存检测到的人物图像（原始方法，生成单独的TSV文件）

        Args:
            detections: 检测结果列表
            output_dir: 输出目录
            base_name: 文件名前缀

        Returns:
            List[str]: 保存的图像路径列表
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        tsv_data = []

        for i, (person_img, bbox) in enumerate(detections):
            if person_img.size == 0:
                continue

            # 生成图片文件名
            image_filename = f"{base_name}_{i:04d}.jpg"
            output_path = os.path.join(output_dir, image_filename)
            
            # 保存图片
            cv2.imwrite(output_path, person_img)
            saved_paths.append(output_path)
            logger.info(f"保存检测到的人物: {output_path}")

            # 将图片编码为base64
            base64_encoded = self._image_to_base64(person_img)
            if base64_encoded:
                tsv_data.append([image_filename, base64_encoded])
            else:
                logger.warning(f"无法编码图片: {output_path}")

        # 保存TSV文件
        if tsv_data:
            tsv_path = os.path.join(output_dir, f"{base_name}_base64.tsv")
            
            with open(tsv_path, 'w', encoding='utf-8') as tsvfile:
                for row in tsv_data:
                    tsvfile.write('\t'.join(row) + '\n')
            
            logger.info(f"保存base64编码数据到: {tsv_path}")
            logger.info(f"共保存 {len(tsv_data)} 张图片的base64编码")

        return saved_paths

    def save_detected_persons_append(self, detections: List[Tuple[np.ndarray, Tuple[int, int, int, int]]], 
                                   output_dir: str, base_name: str = "person",
                                   global_tsv_path: str = "all_detections_base64.tsv") -> List[str]:
        """
        保存检测到的人物图像，并追加到全局TSV文件

        Args:
            detections: 检测结果列表
            output_dir: 输出目录
            base_name: 文件名前缀
            global_tsv_path: 全局TSV文件路径

        Returns:
            List[str]: 保存的图像路径列表
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        new_data = []

        for i, (person_img, bbox) in enumerate(detections):
            if person_img.size == 0:
                continue

            # 生成图片文件名（包含批次信息避免重名）
            image_filename = f"{base_name}_{i:04d}.jpg"
            output_path = os.path.join(output_dir, image_filename)
            
            # 保存图片
            cv2.imwrite(output_path, person_img)
            saved_paths.append(output_path)
            logger.info(f"保存检测到的人物: {output_path}")

            # 将图片编码为base64
            base64_encoded = self._image_to_base64(person_img)
            if base64_encoded:
                new_data.append([image_filename, base64_encoded])
            else:
                logger.warning(f"无法编码图片: {output_path}")

        # 追加数据到全局TSV文件
        if new_data:
            self._append_to_tsv(new_data, global_tsv_path)
            logger.info(f"追加 {len(new_data)} 条数据到全局TSV: {global_tsv_path}")

        return saved_paths

    def _image_to_base64(self, image: np.ndarray) -> str:
        """
        将OpenCV图像转换为base64字符串
        
        Args:
            image: OpenCV图像
            
        Returns:
            str: base64编码的字符串，失败返回空字符串
        """
        try:
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                return base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            return ""
        except Exception as e:
            logger.error(f"图片base64编码失败: {e}")
            return ""

    def _append_to_tsv(self, data: List[List[str]], tsv_path: str):
        """
        追加数据到TSV文件，如果文件不存在则创建
        
        Args:
            data: 要追加的数据列表
            tsv_path: TSV文件路径
        """
        file_exists = os.path.isfile(tsv_path)
        
        try:
            with open(tsv_path, 'a', encoding='utf-8') as tsvfile:
                # 如果文件不存在，不需要写表头，直接写数据
                for row in data:
                    tsvfile.write('\t'.join(row) + '\n')
                
        except Exception as e:
            logger.error(f"写入TSV文件失败: {e}")