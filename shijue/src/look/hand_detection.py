"""
手部关节检测模块
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
from .yolo_detector import YOLODetector


class HandDetector:
    """手部检测器类"""
    
    def __init__(self, yolo_model_path: str = None):
        """
        初始化手部检测器
        
        Args:
            yolo_model_path: YOLO模型路径（可选）
        """
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化手部检测器
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2  # 最多检测2只手
        )
        
        # 初始化YOLO检测器（如果提供路径）
        self.yolo_detector = None
        if yolo_model_path:
            try:
                self.yolo_detector = YOLODetector(yolo_model_path)
            except Exception as e:
                print(f"⚠️ YOLO检测器初始化失败: {e}")
    
    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, List, List]:
        """
        处理单帧图像
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            处理后的图像、手部关键点、YOLO检测结果
        """
        # 转换为RGB格式（MediaPipe需要）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 手部检测
        results = self.hands.process(image_rgb)
        
        # 转换回BGR格式
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        hand_landmarks_list = []
        yolo_detections = []
        
        # 绘制手部关键点
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 提取关键点坐标
                landmarks = self._extract_landmarks(hand_landmarks, image_bgr.shape)
                hand_landmarks_list.append(landmarks)
        
        # YOLO目标检测
        if self.yolo_detector:
            yolo_detections = self.yolo_detector.detect(image_bgr)
            
            # 绘制YOLO检测框
            for detection in yolo_detections:
                x1, y1, x2, y2, conf, class_id = detection
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image_bgr, 
                    f'Class: {class_id}, Conf: {conf:.2f}', 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
        
        return image_bgr, hand_landmarks_list, yolo_detections
    
    def _extract_landmarks(self, hand_landmarks, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """
        提取手部关键点坐标
        
        Args:
            hand_landmarks: MediaPipe手部关键点
            image_shape: 图像尺寸 (height, width, channels)
            
        Returns:
            关键点坐标列表 [(x, y), ...]
        """
        height, width = image_shape[:2]
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))
        
        return landmarks
    
    def get_hand_gesture(self, landmarks: List[Tuple[int, int]]) -> str:
        """
        识别手部手势（简单示例）
        
        Args:
            landmarks: 手部关键点坐标
            
        Returns:
            手势名称
        """
        if len(landmarks) < 21:  # MediaPipe手部检测有21个关键点
            return "Unknown"
        
        # 这里可以添加更复杂的手势识别逻辑
        # 简单示例：检测手掌是否张开
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # 计算手指是否伸展（简化逻辑）
        # 实际应用中需要更复杂的几何计算
        
        return "Hand Detected"
    
    def release(self):
        """释放资源"""
        if self.hands:
            self.hands.close()





def process_camera(
    camera_id: int = 0, 
    yolo_model_path: str = None,
    save_frames: bool = False
) -> None:
    """
    处理摄像头实时流
    
    Args:
        camera_id: 摄像头ID（通常0是默认摄像头）
        yolo_model_path: YOLO模型路径
        save_frames: 是否保存检测帧
    """
    # 初始化检测器
    detector = HandDetector(yolo_model_path)
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头: {camera_id}")
        return
    
    print(f"📹 开始摄像头检测 (ID: {camera_id})")
    print("💡 按 'q' 退出，按 's' 保存当前帧")
    if save_frames:
        print("💾 自动保存模式已启用")
    
    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("⚠️ 无法读取摄像头帧")
                break
            
            # 处理帧
            processed_frame, hand_landmarks, yolo_detections = detector.process_frame(frame)
            
            # 显示处理结果
            cv2.imshow('Real-time Hand Detection + YOLO', processed_frame)
            
            # 自动保存帧（如果启用）
            if save_frames and frame_count % 30 == 0:  # 每30帧保存一次
                filename = f"auto_capture_{frame_count:06d}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"💾 自动保存帧: {filename}")
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 手动保存当前帧
                filename = f"manual_capture_{cv2.getTickCount()}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"💾 手动保存帧: {filename}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print(f"✅ 摄像头检测结束，共处理 {frame_count} 帧")


def process_realsense(
    yolo_model_path: str = None,
    save_frames: bool = False
) -> None:
    """
    处理Intel RealSense D435摄像头实时流
    
    Args:
        yolo_model_path: YOLO模型路径
        save_frames: 是否保存检测帧
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("❌ 未安装pyrealsense2库")
        print("请运行: pip install pyrealsense2")
        return
    
    # 初始化检测器
    detector = HandDetector(yolo_model_path)
    
    # 配置RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用彩色流和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        # 启动管道
        profile = pipeline.start(config)
        print("✅ Intel RealSense D435 摄像头连接成功")
        
        # 获取深度传感器
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"📏 深度比例因子: {depth_scale}")
        
        # 创建深度到彩色的对齐器
        align = rs.align(rs.stream.color)
        
        print("📹 开始RealSense摄像头检测")
        print("💡 按 'q' 退出，按 's' 保存当前帧")
        if save_frames:
            print("💾 自动保存模式已启用")
        
        frame_count = 0
        
        try:
            while True:
                # 等待新的帧
                frames = pipeline.wait_for_frames()
                
                # 对齐深度帧到彩色帧
                aligned_frames = align.process(frames)
                
                # 获取对齐后的帧
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # 处理彩色帧进行手部检测
                processed_frame, hand_landmarks, yolo_detections = detector.process_frame(color_image)
                
                # 在深度图像上绘制手部检测结果
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # 如果有手部检测，在深度图上也绘制
                if hand_landmarks:
                    for landmarks in hand_landmarks:
                        for landmark in landmarks:
                            x, y = int(landmark[0]), int(landmark[1])
                            if 0 <= x < depth_colormap.shape[1] and 0 <= y < depth_colormap.shape[0]:
                                depth_value = depth_image[y, x]
                                if depth_value > 0:
                                    cv2.circle(depth_colormap, (x, y), 3, (0, 255, 0), -1)
                                    cv2.putText(depth_colormap, f"{depth_value}mm", (x+5, y-5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # 显示处理结果
                cv2.imshow('RealSense - Hand Detection + YOLO', processed_frame)
                cv2.imshow('RealSense - Depth Map', depth_colormap)
                
                # 自动保存帧（如果启用）
                if save_frames and frame_count % 30 == 0:  # 每30帧保存一次
                    color_filename = f"realsense_color_{frame_count:06d}.jpg"
                    depth_filename = f"realsense_depth_{frame_count:06d}.jpg"
                    cv2.imwrite(color_filename, processed_frame)
                    cv2.imwrite(depth_filename, depth_colormap)
                    print(f"💾 自动保存帧: {color_filename}, {depth_filename}")
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 手动保存当前帧
                    timestamp = cv2.getTickCount()
                    color_filename = f"realsense_manual_color_{timestamp}.jpg"
                    depth_filename = f"realsense_manual_depth_{timestamp}.jpg"
                    cv2.imwrite(color_filename, processed_frame)
                    cv2.imwrite(depth_filename, depth_colormap)
                    print(f"💾 手动保存帧: {color_filename}, {depth_filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
        
        finally:
            # 停止管道
            pipeline.stop()
            cv2.destroyAllWindows()
            detector.release()
            print(f"✅ RealSense摄像头检测结束，共处理 {frame_count} 帧")
    
    except Exception as e:
        print(f"❌ RealSense摄像头连接失败: {e}")
        print("请检查摄像头连接和驱动安装")
        pipeline.stop()
        detector.release()
