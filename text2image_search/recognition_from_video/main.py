import os
import argparse
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_utils import extract_keyframes, process_image
from detection import PersonDetector

# 配置日志
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(video_path: str, output_dir: str, interval_seconds: int = 2) -> List[str]:
    """
    处理视频文件

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        interval_seconds: 关键帧间隔（秒）

    Returns:
        List[str]: 检测到的人物图像路径列表
    """
    # 创建子目录
    frames_dir = os.path.join(output_dir, "frames")
    persons_dir = os.path.join(output_dir, "persons")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(persons_dir, exist_ok=True)

    # 设置全局TSV文件路径
    global_tsv_path = os.path.join(output_dir, "all_detections_base64.tsv")

    # 提取关键帧
    logger.info(f"开始处理视频: {video_path}")
    frame_paths = extract_keyframes(video_path, frames_dir, interval_seconds)

    # 初始化检测器
    detector = PersonDetector()

    # 检测并保存人物
    all_persons = []
    for frame_path in frame_paths:
        detections = detector.detect_persons(frame_path)
        # 使用新的保存方法，追加到全局TSV
        saved_paths = detector.save_detected_persons_append(
            detections, 
            persons_dir, 
            os.path.basename(frame_path).split('.')[0],
            global_tsv_path
        )
        all_persons.extend(saved_paths)

    logger.info(f"视频处理完成，共检测到 {len(all_persons)} 个人物")
    logger.info(f"所有检测结果已保存到: {global_tsv_path}")
    return all_persons

def process_image_file(image_path: str, output_dir: str) -> List[str]:
    """
    处理图像文件

    Args:
        image_path: 图像文件路径
        output_dir: 输出目录

    Returns:
        List[str]: 检测到的人物图像路径列表
    """
    # 创建子目录
    processed_dir = os.path.join(output_dir, "processed_images")
    persons_dir = os.path.join(output_dir, "persons")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(persons_dir, exist_ok=True)

    # 设置全局TSV文件路径
    global_tsv_path = os.path.join(output_dir, "all_detections_base64.tsv")

    # 处理图像
    logger.info(f"开始处理图像: {image_path}")
    processed_path = process_image(image_path, processed_dir)

    # 初始化检测器
    detector = PersonDetector()

    # 检测并保存人物
    detections = detector.detect_persons(processed_path)
    saved_paths = detector.save_detected_persons_append(
        detections, 
        persons_dir, 
        os.path.basename(image_path).split('.')[0],
        global_tsv_path
    )

    logger.info(f"图像处理完成，共检测到 {len(saved_paths)} 个人物")
    logger.info(f"所有检测结果已保存到: {global_tsv_path}")
    return saved_paths

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="视频/图像中的人物检测工具")
    parser.add_argument("input", help="输入视频或图像文件路径")
    parser.add_argument("-o", "--output", default="output", help="输出目录 (默认为'output')")
    parser.add_argument("-i", "--interval", type=int, default=2,
                        help="视频关键帧间隔秒数 (默认为2秒)")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    # 根据输入类型处理
    input_path = args.input
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(input_path, args.output, args.interval)
    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        process_image_file(input_path, args.output)
    else:
        raise ValueError("不支持的输入文件格式")

if __name__ == "__main__":
    main()