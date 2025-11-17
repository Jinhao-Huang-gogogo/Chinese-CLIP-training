# 视频/图像人物检测应用

这是一个基于YOLOv8的视频和图像中人物检测应用，可以从视频中提取关键帧并检测其中的人物，将检测到的人物保存为单独的图像文件。

## 功能特性

- ✅ 支持视频文件输入（MP4、AVI、MOV、MKV等格式）
- ✅ 支持图像文件输入（JPG、JPEG、PNG、BMP等格式）
- ✅ 自动从视频中提取关键帧
- ✅ 使用YOLOv8进行高精度人物检测
- ✅ 将检测到的人物保存为单独的图像文件
- ✅ 灵活的配置选项

## 安装依赖

```bash
pip install ultralytics opencv-python
```

或者使用项目已有的环境：
```bash
pip install -r requirements.txt
pip install ultralytics
```

## 使用方法

### 命令行使用

```bash
# 处理视频文件
python main.py /path/to/video.mp4 -o output_dir -i 2

# 处理图像文件
python main.py /path/to/image.jpg -o output_dir
```

### 参数说明

- `input`: 输入文件路径（必需）
- `-o, --output`: 输出目录（默认为'output'）
- `-i, --interval`: 视频关键帧间隔秒数（默认为2秒）

### Python API使用

```python
from main import process_video, process_image_file

# 处理视频
person_images = process_video("video.mp4", "output_dir", interval_seconds=2)

# 处理图像
person_images = process_image_file("image.jpg", "output_dir")
```

## 输出结构

处理完成后，输出目录结构如下：

```
output_dir/
├── frames/                 # 提取的关键帧（仅视频处理）
│   ├── keyframe_0000_t0.00s.jpg
│   ├── keyframe_0001_t1.00s.jpg
│   └── ...
├── processed_images/      # 处理的原始图像（仅图像处理）
│   └── input_image.jpg
└── persons/               # 检测到的人物图像
    ├── keyframe_0000_t0.00s_person_0000.jpg
    ├── keyframe_0000_t0.00s_person_0001.jpg
    └── ...
```

## 代码结构

- `main.py` - 主应用程序入口
- `video_utils.py` - 视频处理工具函数
- `detection.py` - 人物检测功能
- `test_app.py` - 功能测试脚本

## 技术细节

- **YOLO模型**: 使用YOLOv8n（nano版本），平衡速度和精度
- **关键帧提取**: 基于时间间隔和运动检测两种方法
- **人物检测**: 只检测'person'类别（class_id=0）
- **置信度阈值**: 默认0.5，可在代码中调整

## 性能优化

- 对于长视频，建议增加关键帧间隔（-i参数）
- 可以使用更大的YOLO模型（如yolov8s.pt、yolov8m.pt）提高检测精度
- 支持GPU加速（如果系统有CUDA）

## 示例

```bash
# 处理视频，每5秒提取一帧
python main.py sample_video.mp4 -o results -i 5

# 处理单张图像
python main.py photo.jpg -o detected_persons
```

## 注意事项

- 首次运行会自动下载YOLOv8模型（约6.2MB）
- 确保有足够的磁盘空间存储输出文件
- 处理大视频文件可能需要较长时间