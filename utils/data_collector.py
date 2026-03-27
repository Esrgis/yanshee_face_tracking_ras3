import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

class DataCollector:
    def __init__(self, output_dir="../data/videos", fps=25, resolution=(640, 480)):
        """
        Khởi tạo bộ thu thập dữ liệu từ camera
        
        Args:
            output_dir: Đường dẫn thư mục lưu video
            fps: Số frame trên giây
            resolution: Độ phân giải (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Khởi tạo camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Kiểm tra camera
        if not self.cap.isOpened():
            raise RuntimeError("Không thể mở camera")
        
        # Biến trạng thái
        self.is_recording = False
        self.recording_type = None  # 'slow', 'fast', 'sudden'
        self.frame_count = 0
        self.recorded_frames = 0
        
        # Video writer
        self.out = None
        self.video_filename = None
        
        # Codec cho AVI (MJPG - 4 ký tự, không phải MJPEG)
        self.codec = cv2.VideoWriter_fourcc(*'MJPG')
        
    def get_timestamp(self):
        """Lấy timestamp hiện tại"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def start_recording(self, recording_type):
        """
        Bắt đầu ghi video
        
        Args:
            recording_type: 'slow', 'fast', hoặc 'sudden'
        """
        if self.is_recording:
            self.stop_recording()
        
        self.recording_type = recording_type
        self.recorded_frames = 0
        
        # Tạo tên file
        timestamp = self.get_timestamp()
        type_name = {
            'slow': 'slow_motion',
            'fast': 'fast_motion',
            'sudden': 'sudden_motion'
        }
        
        self.video_filename = self.output_dir / f"{type_name[recording_type]}_{timestamp}.avi"
        
        # Khởi tạo video writer
        self.out = cv2.VideoWriter(
            str(self.video_filename),
            self.codec,
            self.fps,
            (self.width, self.height)
        )
        
        if not self.out.isOpened():
            print(f"Lỗi: Không thể tạo file video {self.video_filename}")
            self.out = None
            return False
        
        self.is_recording = True
        print(f"Bắt đầu ghi video: {self.recording_type} -> {self.video_filename}")
        return True
    
    def stop_recording(self):
        """Dừng ghi video"""
        if self.is_recording and self.out is not None:
            self.out.release()
            print(f"Đã dừng ghi video: {self.recording_type} ({self.recorded_frames} frames)")
            self.is_recording = False
            self.recording_type = None
            self.out = None
            self.recorded_frames = 0  # Reset frame counter
    
    def draw_info(self, frame):
        """
        Vẽ thông tin lên frame
        
        Args:
            frame: Frame từ camera
            
        Returns:
            Frame có thêm thông tin
        """
        frame = frame.copy()
        
        # Thông tin số frame được thu thập
        cv2.putText(
            frame,
            f"Frames Collected: {self.recorded_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Thông tin ghi video
        if self.is_recording:
            recording_text = f"RECORDING: {self.recording_type.upper()} ({self.recorded_frames})"
            color = (0, 0, 255)  # Đỏ cho recording
            cv2.putText(
                frame,
                recording_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        else:
            cv2.putText(
                frame,
                "Not Recording",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )
        
        # Hướng dẫn phím tắt
        instructions = [
            "1: Slow Motion | 2: Fast Motion | 3: Sudden Motion",
            "S: Stop Recording | Q: Close"
        ]
        
        for idx, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (10, self.height - 20 - (idx * 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def run(self):
        """Chạy chương trình thu thập dữ liệu"""
        print("Giao diện camera đang chạy...")
        print("Phím tắt:")
        print("  1: Bắt đầu lưu video chuyển động chậm")
        print("  2: Bắt đầu lưu video chuyển động nhanh")
        print("  3: Bắt đầu lưu video chuyển động đột ngột")
        print("  S: Dừng lưu video")
        print("  Q: Thoát chương trình")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Lỗi: Không thể đọc frame từ camera")
                    break
                
                self.frame_count += 1
                
                # Ghi frame nếu đang ghi video
                if self.is_recording and self.out is not None:
                    self.out.write(frame)
                    self.recorded_frames += 1
                
                # Vẽ thông tin lên frame
                display_frame = self.draw_info(frame)
                
                # Hiển thị frame
                cv2.imshow("Camera - Data Collector", display_frame)
                
                # Xử lý input từ bàn phím
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('1'):
                    self.start_recording('slow')
                elif key == ord('2'):
                    self.start_recording('fast')
                elif key == ord('3'):
                    self.start_recording('sudden')
                elif key == ord('s') or key == ord('S'):
                    self.stop_recording()
                elif key == ord('q') or key == ord('Q'):
                    print("Đang thoát...")
                    break
        
        except KeyboardInterrupt:
            print("\nGương program bị gián đoạn")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Làm sạch tài nguyên"""
        if self.is_recording:
            self.stop_recording()
        
        if self.out is not None:
            self.out.release()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"Tổng số frame thu thập: {self.frame_count}")
        print(f"Video được lưu tại: {self.output_dir.absolute()}")
        print("Chương trình đã kết thúc")


def main():
    """Hàm chính"""
    try:
        collector = DataCollector(
            output_dir="../data/videos",
            fps=25,
            resolution=(640, 480)
        )
        collector.run()
    except Exception as e:
        print(f"Lỗi: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
