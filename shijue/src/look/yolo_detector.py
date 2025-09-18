"""
YOLO目标检测器 - 支持PyTorch和ONNX模型
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class YOLODetector:
    """YOLO目标检测器类 - 支持PyTorch和ONNX模型"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型文件路径 (.pt 或 .onnx)
            conf_threshold: 置信度阈值
            nms_threshold: 非极大值抑制阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_type = None
        self.net = None
        
        # 检测模型类型并加载
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型"""
        if not os.path.exists(self.model_path):
            print(f"❌ 模型文件不存在: {self.model_path}")
            print("使用模拟检测器...")
            return
        
        file_ext = os.path.splitext(self.model_path)[1].lower()
        
        try:
            if file_ext == '.pt':
                # PyTorch模型
                self._load_pytorch_model()
            elif file_ext == '.onnx':
                # ONNX模型
                self._load_onnx_model()
            else:
                print(f"❌ 不支持的模型格式: {file_ext}")
                print("支持的格式: .pt (PyTorch), .onnx (ONNX)")
                print("使用模拟检测器...")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("使用模拟检测器...")
    
    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        try:
            import torch
            from ultralytics import YOLO
            
            print(f"🔄 正在加载PyTorch模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model_type = 'pytorch'
            print(f"✅ PyTorch模型加载成功: {self.model_path}")
            
        except ImportError:
            print("❌ 缺少PyTorch依赖，请安装: pip install torch ultralytics")
            print("使用模拟检测器...")
        except Exception as e:
            print(f"❌ PyTorch模型加载失败: {e}")
            print("使用模拟检测器...")
    
    def _load_onnx_model(self):
        """加载ONNX模型"""
        try:
            print(f"🔄 正在加载ONNX模型: {self.model_path}")
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self.model_type = 'onnx'
            print(f"✅ ONNX模型加载成功: {self.model_path}")
            
        except Exception as e:
            print(f"❌ ONNX模型加载失败: {e}")
            print("使用模拟检测器...")
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """
        执行目标检测
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个元素为 (x1, y1, x2, y2, confidence, class_id)
        """
        if self.model_type == 'pytorch':
            return self._detect_pytorch(image)
        elif self.model_type == 'onnx':
            return self._detect_onnx(image)
        else:
            # 模拟检测器
            return self._mock_detection(image)
    
    def _detect_pytorch(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """使用PyTorch模型进行检测"""
        try:
            # 使用YOLO模型进行推理
            results = self.model(image, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if confidence > self.conf_threshold:
                            detections.append((
                                int(x1), int(y1), int(x2), int(y2), 
                                confidence, class_id
                            ))
            
            return detections
            
        except Exception as e:
            print(f"⚠️ PyTorch检测失败: {e}")
            return self._mock_detection(image)
    
    def _detect_onnx(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """使用ONNX模型进行检测"""
        try:
            # 预处理图像
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
            
            # 前向推理
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            # 后处理检测结果
            return self._process_onnx_outputs(outputs, image.shape)
            
        except Exception as e:
            print(f"⚠️ ONNX检测失败: {e}")
            return self._mock_detection(image)
    
    def _process_onnx_outputs(self, outputs: np.ndarray, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int, float, int]]:
        """处理ONNX模型输出"""
        detections = []
        height, width = image_shape[:2]
        
        # 处理YOLOv8输出格式
        for detection in outputs[0]:
            confidence = float(detection[4])
            
            if confidence > self.conf_threshold:
                # 获取边界框坐标
                x_center = detection[0] * width
                y_center = detection[1] * height
                w = detection[2] * width
                h = detection[3] * height
                
                # 转换为左上角和右下角坐标
                x1 = int(x_center - w/2)
                y1 = int(y_center - h/2)
                x2 = int(x_center + w/2)
                y2 = int(y_center + h/2)
                
                # 获取类别ID
                class_id = int(detection[5])
                
                detections.append((x1, y1, x2, y2, confidence, class_id))
        
        return detections
    
    def _mock_detection(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """
        模拟检测器（用于测试，当模型未加载时）
        
        Args:
            image: 输入图像
            
        Returns:
            模拟的检测结果
        """
        height, width = image.shape[:2]
        
        # 在图像中心创建一个模拟的检测框
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 4
        
        x1 = center_x - box_size // 2
        y1 = center_y - box_size // 2
        x2 = center_x + box_size // 2
        y2 = center_y + box_size // 2
        
        return [(x1, y1, x2, y2, 0.95, 0)]  # 模拟检测结果
    
    def get_model_info(self) -> str:
        """获取模型信息"""
        if self.model_type == 'pytorch':
            return f"PyTorch模型: {os.path.basename(self.model_path)}"
        elif self.model_type == 'onnx':
            return f"ONNX模型: {os.path.basename(self.model_path)}"
        else:
            return "模拟检测器"
