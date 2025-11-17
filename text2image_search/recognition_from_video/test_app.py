#!/usr/bin/env python3
"""
测试应用程序功能
"""
import os
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_utils import extract_keyframes, process_image
from detection import PersonDetector

def test_video_utils():
    """测试视频工具功能"""
    print("测试视频工具功能...")

    # 创建一个测试视频
    test_video_path = "/data/jinhaohuang/Chinese-CLIP/text2image_search/recognition_from_video/test_data/test_video.avi"
    output_dir = "/data/jinhaohuang/Chinese-CLIP/text2image_search/recognition_from_video/test_data/frames"

    # 创建测试视频（如果有OpenCV的测试视频可以跳过这一步）
    if not os.path.exists(test_video_path):
        print("创建测试视频...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(test_video_path, fourcc, 20.0, (640, 480))

        for i in range(100):  # 100帧测试视频
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    # 测试关键帧提取
    try:
        frame_paths = extract_keyframes(test_video_path, output_dir, interval_seconds=1)
        print(f"✓ 成功提取 {len(frame_paths)} 个关键帧")
        return True
    except Exception as e:
        print(f"✗ 关键帧提取失败: {e}")
        return False

def test_detection():
    """测试人物检测功能"""
    print("测试人物检测功能...")

    # 创建一个测试图像
    test_image_path = "/data/jinhaohuang/Chinese-CLIP/text2image_search/recognition_from_video/test_data/test_image.jpg"
    output_dir = "/data/jinhaohuang/Chinese-CLIP/text2image_search/recognition_from_video/test_data/detections"

    # 创建测试图像
    if not os.path.exists(test_image_path):
        print("创建测试图像...")
        # 创建一个包含人物的简单图像
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加一些矩形模拟人物
        cv2.rectangle(image, (100, 100), (200, 300), (255, 255, 255), -1)  # 白色矩形
        cv2.rectangle(image, (400, 150), (500, 350), (200, 200, 200), -1)  # 灰色矩形
        cv2.imwrite(test_image_path, image)

    # 测试人物检测
    try:
        detector = PersonDetector()
        detections = detector.detect_persons(test_image_path)
        saved_paths = detector.save_detected_persons(detections, output_dir)
        print(f"✓ 成功检测到 {len(detections)} 个人物")
        return True
    except Exception as e:
        print(f"✗ 人物检测失败: {e}")
        return False

def test_main_functionality():
    """测试主要功能"""
    print("测试主要功能...")

    # 测试图像处理
    test_image_path = "/data/jinhaohuang/Chinese-CLIP/text2image_search/recognition_from_video/test_data/test_main.jpg"
    output_dir = "/data/jinhaohuang/Chinese-CLIP/text2image_search/recognition_from_video/test_data/main_output"

    # 创建测试图像
    if not os.path.exists(test_image_path):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (200, 300), (255, 255, 255), -1)
        cv2.imwrite(test_image_path, image)

    try:
        # 测试图像处理
        from main import process_image_file
        result = process_image_file(test_image_path, output_dir)
        print(f"✓ 主要功能测试成功，检测到 {len(result)} 个人物")
        return True
    except Exception as e:
        print(f"✗ 主要功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试应用程序功能...\n")

    # 运行测试
    test1 = test_video_utils()
    test2 = test_detection()
    test3 = test_main_functionality()

    print("\n测试结果:")
    print(f"视频工具测试: {'通过' if test1 else '失败'}")
    print(f"人物检测测试: {'通过' if test2 else '失败'}")
    print(f"主要功能测试: {'通过' if test3 else '失败'}")

    if test1 and test2 and test3:
        print("\n🎉 所有测试通过！应用程序功能正常。")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")