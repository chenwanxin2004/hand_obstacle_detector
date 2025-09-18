"""
手部检测和YOLO目标检测项目

这是一个集成了MediaPipe手部检测和YOLO目标检测的Python项目。
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 在这里导入主要的模块
from .main import main
from .hand_detection import HandDetector, process_camera, process_realsense
from .yolo_detector import YOLODetector

__all__ = [
    "main", 
    "HandDetector", 
    "process_camera", 
    "process_realsense", 
    "YOLODetector"
]
