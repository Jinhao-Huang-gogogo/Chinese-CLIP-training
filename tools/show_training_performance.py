import re
import matplotlib.pyplot as plt
import os

def parse_training_log(log_file_path):
    """
    解析训练日志文件，提取每个epoch的loss、acc和logit_scale信息
    """
    epochs = []
    steps = []
    losses = []
    img2text_acc = []
    text2img_acc = []
    logit_scales = []
    
    # 正则表达式匹配验证结果行
    pattern = r'Validation Result \(epoch (\d+) @ (\d+) steps\) \| Valid Loss: ([\d.]+) \| Image2Text Acc: ([\d.]+) \| Text2Image Acc: ([\d.]+) \| logit_scale: ([\d.]+)'
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = float(match.group(3))
                acc_img2text = float(match.group(4))
                acc_text2img = float(match.group(5))
                logit_scale = float(match.group(6))
                
                epochs.append(epoch)
                steps.append(step)
                losses.append(loss)
                img2text_acc.append(acc_img2text)
                text2img_acc.append(acc_text2img)
                logit_scales.append(logit_scale)
                
                print(f"Epoch {epoch} Step {step}: Loss={loss:.4f}, "
                      f"Img2Text Acc={acc_img2text:.2f}%, "
                      f"Text2Img Acc={acc_text2img:.2f}%, "
                      f"logit_scale={logit_scale:.3f}")
    
    return epochs, steps, losses, img2text_acc, text2img_acc, logit_scales

def plot_training_metrics(epochs, steps, losses, img2text_acc, text2img_acc, logit_scales, save_path='training_metrics.png'):
    """
    绘制训练指标变化图 - 双X轴显示step和epoch
    """
    # 创建2x2的子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 为每个子图创建双X轴
    ax1_epoch = ax1.twiny()
    ax2_epoch = ax2.twiny()
    ax3_epoch = ax3.twiny()
    ax4_epoch = ax4.twiny()
    
    # 1. 绘制Loss变化
    ax1.plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss vs Step/Epoch')
    ax1.grid(True, alpha=0.3)
    
    # 设置epoch轴
    ax1_epoch.set_xlabel('Epoch')
    ax1_epoch.set_xlim(ax1.get_xlim())
    epoch_positions = [steps[0]] + [steps[i] for i in range(len(steps)) if epochs[i] != epochs[i-1]] if len(steps) > 1 else [steps[0]]
    epoch_labels = [str(epochs[0])] + [str(epochs[i]) for i in range(len(epochs)) if epochs[i] != epochs[i-1]] if len(epochs) > 1 else [str(epochs[0])]
    ax1_epoch.set_xticks(epoch_positions)
    ax1_epoch.set_xticklabels(epoch_labels)
    
    # 2. 绘制Accuracy变化
    ax2.plot(steps, img2text_acc, 'g-', linewidth=2, marker='s', markersize=4, label='Image2Text')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Image2Text Accuracy vs Step/Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax2_epoch.set_xlabel('Epoch')
    ax2_epoch.set_xlim(ax2.get_xlim())
    ax2_epoch.set_xticks(epoch_positions)
    ax2_epoch.set_xticklabels(epoch_labels)
    
    # 3. 绘制两种Accuracy对比
    ax3.plot(steps, text2img_acc, 'r-', linewidth=2, marker='^', markersize=4, label='Text2Image')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Text2Image Accuracy vs Step/Epoch')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax3_epoch.set_xlabel('Epoch')
    ax3_epoch.set_xlim(ax3.get_xlim())
    ax3_epoch.set_xticks(epoch_positions)
    ax3_epoch.set_xticklabels(epoch_labels)
    
    # 4. 绘制logit_scale变化
    ax4.plot(steps, logit_scales, 'purple', linewidth=2, marker='d', markersize=4)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Logit Scale')
    ax4.set_title('Logit Scale vs Step/Epoch')
    ax4.grid(True, alpha=0.3)
    
    ax4_epoch.set_xlabel('Epoch')
    ax4_epoch.set_xlim(ax4.get_xlim())
    ax4_epoch.set_xticks(epoch_positions)
    ax4_epoch.set_xticklabels(epoch_labels)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {save_path}")
    plt.show()
    
    return fig
import glob

def main():
    # 配置日志文件路径和输出图片路径
    import argparse
    parser = argparse.ArgumentParser(description="显示训练性能图表")
    parser.add_argument('--log_file', type=str, help='训练日志文件路径或包含log文件的目录路径')
    parser.add_argument('--log_dir', type=str, default="/data/jinhaohuang/Chinese-CLIP/experiments/car-V0.1-people_V1.1_finetune_vit-l-14_roberta-base-small-lr", help='包含log文件的目录路径')
    args = parser.parse_args()

    # 确定log文件列表
    log_files = []
    
    if args.log_file:
        # 如果指定了具体的log文件
        if os.path.isfile(args.log_file):
            log_files = [args.log_file]
        elif os.path.isdir(args.log_file):
            # 如果指定的是目录，查找该目录下的log文件
            log_files = glob.glob(os.path.join(args.log_file, "*.log"))
        else:
            print(f"错误: 指定的路径不存在: {args.log_file}")
            return
    else:
        # 如果没有指定log_file，使用log_dir
        if os.path.isdir(args.log_dir):
            log_files = glob.glob(os.path.join(args.log_dir, "*.log"))
        else:
            print(f"错误: 目录不存在: {args.log_dir}")
            return

    # 如果没有找到log文件
    if not log_files:
        print(f"在指定路径中未找到任何.log文件: {args.log_file or args.log_dir}")
        return

    print(f"找到 {len(log_files)} 个log文件:")
    for log_file in log_files:
        print(f"  - {log_file}")

    # 处理每个log文件
    for log_file in log_files:
        try:
            print(f"\n正在处理: {log_file}")
            
            # 解析日志
            epochs, steps, losses, img2text_acc, text2img_acc, logit_scales = parse_training_log(log_file)
            
            if not epochs:
                print(f"  未找到有效的训练数据，跳过该文件")
                continue
            
            print(f"  成功解析 {len(epochs)} 个训练数据点")
            print(f"  Epoch范围: {min(epochs)} - {max(epochs)}")
            print(f"  Step范围: {min(steps)} - {max(steps)}")
            
            # 生成输出图片路径
            output_image = log_file.replace('.log', '.png')
            
            # 绘制图表
            plot_training_metrics(epochs, steps, losses, img2text_acc, text2img_acc, logit_scales, output_image)
            
            print(f"  图表已保存: {output_image}")
            
        except Exception as e:
            print(f"  处理文件时出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()