"""
手部检测使用示例
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.project_name.hand_detection import HandDetector, process_video_file, process_camera


def example_hand_detection():
    """手部检测基本使用示例"""
    print("🤖 手部检测示例")
    print("=" * 40)
    
    # 创建手部检测器（不包含YOLO）
    detector = HandDetector()
    
    print("✅ 手部检测器创建成功")
    print("💡 使用以下命令运行完整功能:")
    print("   python src/project_name/main.py --mode camera")
    print("   python src/project_name/main.py --mode video --input your_video.mp4")
    
    # 释放资源
    detector.release()


def example_video_processing():
    """视频处理示例"""
    print("\n🎬 视频处理示例")
    print("=" * 40)
    
    # 示例视频路径（请替换为实际路径）
    video_path = "path/to/your/video.mp4"
    output_path = "output_processed.mp4"
    
    print(f"📁 输入视频: {video_path}")
    print(f"💾 输出视频: {output_path}")
    print("💡 要处理视频，请运行:")
    print(f"   python src/project_name/main.py --mode video --input {video_path} --output {output_path}")


def example_camera_processing():
    """摄像头处理示例"""
    print("\n📹 摄像头处理示例")
    print("=" * 40)
    
    print("💡 要启动摄像头检测，请运行:")
    print("   python src/project_name/main.py --mode camera")
    print("   或者指定特定摄像头:")
    print("   python src/project_name/main.py --mode camera --camera-id 1")


def example_with_yolo():
    """包含YOLO检测的示例"""
    print("\n🤖 YOLO检测示例")
    print("=" * 40)
    
    yolo_model_path = "path/to/yolov8n.onnx"
    
    print(f"🤖 YOLO模型: {yolo_model_path}")
    print("💡 要同时使用手部检测和YOLO检测，请运行:")
    print(f"   python src/project_name/main.py --mode camera --yolo-model {yolo_model_path}")
    print(f"   python src/project_name/main.py --mode video --input video.mp4 --yolo-model {yolo_model_path}")


def main():
    """主函数"""
    print("🚀 手部检测和YOLO目标检测项目示例")
    print("=" * 60)
    
    # 运行各种示例
    example_hand_detection()
    example_video_processing()
    example_camera_processing()
    example_with_yolo()
    
    print("\n" + "=" * 60)
    print("🎯 快速开始:")
    print("1. 安装依赖: uv sync")
    print("2. 摄像头检测: python src/project_name/main.py")
    print("3. 视频处理: python src/project_name/main.py --mode video --input your_video.mp4")
    print("4. 查看帮助: python src/project_name/main.py --help")


if __name__ == "__main__":
    main()
