import cv2
import os
import numpy as np
from typing import List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_keyframes(video_path: str, output_dir: str, interval_seconds: int = 2) -> List[str]:
    """
    从视频中提取关键帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        interval_seconds: 关键帧间隔（秒）

    Returns:
        List[str]: 提取的关键帧文件路径列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"视频信息 - FPS: {fps:.2f}, 总帧数: {total_frames}, 时长: {duration:.2f}秒")

    frame_interval = int(fps * interval_seconds)
    if frame_interval == 0:
        frame_interval = 1

    keyframe_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔interval_seconds秒保存一帧
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps if fps > 0 else frame_count
            filename = f"keyframe_{saved_count:04d}_t{timestamp:.2f}s.jpg"
            output_path = os.path.join(output_dir, filename)

            # 保存关键帧
            cv2.imwrite(output_path, frame)
            keyframe_paths.append(output_path)
            saved_count += 1

            logger.info(f"保存关键帧: {filename}")

        frame_count += 1

    cap.release()
    logger.info(f"总共提取了 {saved_count} 个关键帧")

    return keyframe_paths

def extract_frames_by_motion(video_path: str, output_dir: str, motion_threshold: float = 1000.0) -> List[str]:
    """
    基于运动检测提取关键帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        motion_threshold: 运动检测阈值

    Returns:
        List[str]: 提取的关键帧文件路径列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"基于运动检测提取关键帧 - FPS: {fps:.2f}, 总帧数: {total_frames}")

    keyframe_paths = []
    prev_frame = None
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is not None:
            # 计算帧间差异
            frame_diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

            # 计算运动量
            motion_amount = np.sum(thresh) / 255

            # 如果运动量超过阈值，保存为关键帧
            if motion_amount > motion_threshold:
                timestamp = frame_count / fps if fps > 0 else frame_count
                filename = f"motion_frame_{saved_count:04d}_t{timestamp:.2f}s.jpg"
                output_path = os.path.join(output_dir, filename)

                cv2.imwrite(output_path, frame)
                keyframe_paths.append(output_path)
                saved_count += 1

                logger.info(f"检测到运动 ({motion_amount:.2f})，保存帧: {filename}")

        prev_frame = gray.copy()
        frame_count += 1

    cap.release()
    logger.info(f"基于运动检测提取了 {saved_count} 个关键帧")

    return keyframe_paths

def process_image(image_path: str, output_dir: str) -> str:
    """
    处理单张图像，直接复制到输出目录

    Args:
        image_path: 图像文件路径
        output_dir: 输出目录

    Returns:
        str: 输出图像路径
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 生成输出文件名
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"input_{name}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    # 保存图像
    cv2.imwrite(output_path, image)
    logger.info(f"处理图像: {filename} -> {output_filename}")

    return output_path