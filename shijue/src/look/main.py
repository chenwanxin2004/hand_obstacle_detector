#!/usr/bin/env python3
"""
主程序入口文件 - 手部检测和YOLO目标检测（摄像头模式）
支持Intel RealSense D435摄像头
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from .hand_detection import process_camera, process_realsense

# 加载环境变量
load_dotenv()


def main() -> None:
    """
    主函数
    """
    print("🤖 手部检测和YOLO目标检测系统")
    print("📹 摄像头模式 - 支持Intel RealSense D435")
    print("=" * 60)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="手部检测和YOLO目标检测系统（摄像头模式）")
    parser.add_argument(
        "--camera-type", 
        choices=["default", "realsense"], 
        default="default",
        help="摄像头类型：default(默认摄像头) 或 realsense(Intel RealSense D435)"
    )
    parser.add_argument(
        "--camera-id", 
        type=int, 
        default=0,
        help="默认摄像头ID (默认: 0)"
    )
    parser.add_argument(
        "--yolo-model", 
        type=str, 
        default=str(Path(__file__).parent / "yolov8n.pt"),
        help="YOLO模型文件路径"
    )
    parser.add_argument(
        "--save-frames", 
        action="store_true",
        help="是否保存检测帧"
    )
    
    args = parser.parse_args()
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ 需要Python 3.8或更高版本，当前版本: {python_version.major}.{python_version.minor}")
        return
    
    print(f"✅ Python版本检查通过: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"📁 项目根目录: {project_root}")
    
    # 检查环境变量
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    print(f"🔧 调试模式: {debug_mode}")
    print(f"📝 日志级别: {log_level}")
    
    # 检查YOLO模型文件
    yolo_model_path = args.yolo_model
    if not Path(yolo_model_path).exists():
        print(f"❌ YOLO模型文件不存在: {yolo_model_path}")
        print("请确保模型文件已下载到正确位置")
        return
    
    print(f"✅ 找到YOLO模型: {yolo_model_path}")
    
    # 根据摄像头类型运行相应的功能
    if args.camera_type == "realsense":
        print(f"📹 Intel RealSense D435 摄像头模式")
        print("🔍 正在检测RealSense摄像头...")
        
        # 处理RealSense摄像头
        process_realsense(
            yolo_model_path=yolo_model_path,
            save_frames=args.save_frames
        )
    
    else:
        print(f"📹 默认摄像头模式")
        print(f"📷 摄像头ID: {args.camera_id}")
        
        # 处理默认摄像头
        process_camera(
            camera_id=args.camera_id,
            yolo_model_path=yolo_model_path,
            save_frames=args.save_frames
        )
    
    print("\n🎉 程序执行完成!")


if __name__ == "__main__":
    main()
