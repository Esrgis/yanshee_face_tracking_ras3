"""
Face Detector - Sử dụng YOLOv8 để phát hiện khuôn mặt từ camera và video
Lưu annotations theo định dạng YOLO1.1 cho CVAT
"""
import cv2
from ultralytics import YOLO
from pathlib import Path
import json


class FaceDetector:
    def __init__(self, model_path="models/yolov8n-face-lindevs.pt", conf_threshold=0.5):
        """
        Khởi tạo Face Detector
        
        Args:
            model_path: Đường dẫn đến model YOLO
            conf_threshold: Ngưỡng confidence (0-1)
        """
        self.conf_threshold = float(conf_threshold)
        
        # Xác định đường dẫn model từ vị trí file hiện tại
        current_file_dir = Path(__file__).parent.absolute()
        project_root = current_file_dir.parent
        
        # Nếu model_path là tương đối, tính từ thư mục gốc dự án
        if not Path(model_path).is_absolute():
            model_path = project_root / model_path
        
        print(f"[FACE DETECTOR] Đang tải model: {model_path}")
        
        try:
            self.yolo = YOLO(str(model_path))
            print(f"[FACE DETECTOR] ✓ Model tải thành công")
        except Exception as e:
            print(f"[FACE DETECTOR] ✗ Lỗi tải model: {e}")
            raise e
    
    def detect_faces(self, frame):
        """
        Phát hiện khuôn mặt trong frame
        
        Args:
            frame: Frame từ camera
            
        Returns:
            List của bounding boxes: [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.yolo.predict(frame, verbose=False)
        
        faces = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            
            # Chỉ giữ những detection có confidence >= threshold
            if conf >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2, y2, conf))
        
        return faces
    
    def convert_to_yolo_format(self, bbox, frame_width, frame_height):
        """
        Chuyển đổi bounding box từ pixel sang YOLO1.1 format (normalized)
        
        YOLO1.1 format: class_id center_x center_y width height
        Tất cả giá trị được normalize (0-1)
        
        Args:
            bbox: (x1, y1, x2, y2) trong pixel
            frame_width: Chiều rộng frame
            frame_height: Chiều cao frame
            
        Returns:
            String: "class_id center_x center_y width height"
        """
        x1, y1, x2, y2 = bbox
        
        # Tính center và width/height
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize
        center_x /= frame_width
        center_y /= frame_height
        width /= frame_width
        height /= frame_height
        
        # Class ID 0 cho face
        class_id = 0
        
        return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
    
    def save_yolo_annotations(self, frame_path, faces, frame_width, frame_height):
        """
        Lưu annotations theo định dạng YOLO1.1
        
        Args:
            frame_path: Đường dẫn file ảnh frame
            faces: List bounding boxes [(x1, y1, x2, y2, conf), ...]
            frame_width: Chiều rộng frame
            frame_height: Chiều cao frame
        """
        # Tạo file .txt tương ứng
        txt_path = frame_path.with_suffix('.txt')
        
        with open(txt_path, 'w') as f:
            for x1, y1, x2, y2, conf in faces:
                yolo_line = self.convert_to_yolo_format((x1, y1, x2, y2), frame_width, frame_height)
                f.write(yolo_line + '\n')
    
    def draw_faces(self, frame, faces):
        """
        Vẽ bounding box và thông tin lên frame
        
        Args:
            frame: Frame gốc
            faces: List bounding boxes
            
        Returns:
            Frame có vẽ bounding boxes
        """
        frame = frame.copy()
        
        for idx, (x1, y1, x2, y2, conf) in enumerate(faces):
            # Vẽ bounding box
            color = (0, 255, 0)  # Xanh
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Vẽ label với confidence
            label = f"Face {idx + 1}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Khung nền cho label
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0] + 5, y1), color, -1)
            
            # Vẽ text
            cv2.putText(frame, label, (x1 + 3, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Hiển thị số lượng khuôn mặt được phát hiện
        face_count_text = f"Faces detected: {len(faces)}"
        cv2.putText(frame, face_count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Thông tin format
        format_text = "Format: YOLO1.1 (Normalized)"
        cv2.putText(frame, format_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        
        # Hướng dẫn phím tắt
        instructions = [
            "Press 'q' to quit"
        ]
        for idx, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 20 - (idx * 25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_on_camera(self):
        """
        Chạy face detection từ camera
        """
        print("[FACE DETECTOR] Đang kết nối camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[FACE DETECTOR] ✗ Không thể mở camera")
            return
        
        # Cấu hình camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[FACE DETECTOR] ✓ Camera đã kết nối")
        print("[FACE DETECTOR] Nhấn 'q' để thoát")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("[FACE DETECTOR] ✗ Lỗi đọc frame từ camera")
                    break
                
                # Phát hiện khuôn mặt
                faces = self.detect_faces(frame)
                
                # Vẽ bounding boxes
                display_frame = self.draw_faces(frame, faces)
                
                # Hiển thị
                cv2.imshow("Face Detection - Camera Feed", display_frame)
                
                # Xử lý input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("[FACE DETECTOR] Đang thoát...")
                    break
        
        except KeyboardInterrupt:
            print("\n[FACE DETECTOR] Được gián đoạn bởi người dùng")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[FACE DETECTOR] Camera đã được đóng")
    
    def run_on_video(self, video_path):
        """
        Chạy face detection trên file video
        
        Args:
            video_path: Đường dẫn đến file video
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"[FACE DETECTOR] ✗ Không thể tìm video: {video_path}")
            return
        
        print(f"[FACE DETECTOR] Đang mở video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"[FACE DETECTOR] ✗ Không thể mở video: {video_path}")
            return
        
        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[FACE DETECTOR] ✓ Video info: {width}x{height} @ {fps}fps ({total_frames} frames)")
        
        # Tạo folder labels cho video này
        current_file_dir = Path(__file__).parent.absolute()
        project_root = current_file_dir.parent
        data_dir = project_root / "data"
        
        # Tạo tên folder labels từ tên video (bỏ extension)
        video_name = video_path.stem
        labels_dir = data_dir / f"labels_yolo_{video_name}"
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[FACE DETECTOR] ✓ Labels sẽ được lưu tại: {labels_dir}")
        
        # Tạo metadata file
        metadata = {
            "video_path": str(video_path),
            "video_name": video_name,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "format": "YOLO1.1",
            "classes": {
                "0": "face"
            }
        }
        
        with open(labels_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("[FACE DETECTOR] Nhấn 'q' để dừng, 'p' để pause/resume")
        
        frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("[FACE DETECTOR] ✓ Đã hoàn thành video")
                        break
                    
                    frame_count += 1
                
                # Phát hiện khuôn mặt
                faces = self.detect_faces(frame)
                
                # Lưu annotations YOLO1.1
                frame_filename = f"{video_name}_{frame_count:06d}"
                frame_path = labels_dir / frame_filename
                self.save_yolo_annotations(frame_path, [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in faces], width, height)
                
                # Vẽ bounding boxes
                display_frame = self.draw_faces(frame, faces)
                
                # Thêm thông tin progress
                progress_text = f"Frame: {frame_count}/{total_frames} | Faces: {len(faces)}"
                cv2.putText(display_frame, progress_text, (10, display_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Hiển thị labels được lưu
                save_status = f"✓ Labels: {labels_dir.name}"
                cv2.putText(display_frame, save_status, (10, display_frame.shape[0] - 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Hiển thị
                cv2.imshow("Face Detection - Video", display_frame)
                
                # Xử lý input
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    print("[FACE DETECTOR] Đang thoát...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    status = "Paused" if paused else "Running"
                    print(f"[FACE DETECTOR] {status}")
        
        except KeyboardInterrupt:
            print("\n[FACE DETECTOR] Được gián đoạn bởi người dùng")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[FACE DETECTOR] Video đã được đóng")
            print(f"[FACE DETECTOR] Tổng frames xử lý: {frame_count}")
            print(f"[FACE DETECTOR] ✓ Annotations được lưu tại: {labels_dir}")
    
    def process_all_videos_in_data(self):
        """
        Xử lý tất cả các video .avi trong data/videos/
        """
        current_file_dir = Path(__file__).parent.absolute()
        project_root = current_file_dir.parent
        videos_dir = project_root / "data" / "videos"
        
        if not videos_dir.exists():
            print(f"[FACE DETECTOR] ✗ Thư mục không tồn tại: {videos_dir}")
            return
        
        # Tìm tất cả file .avi
        avi_files = list(videos_dir.glob("*.avi"))
        
        if not avi_files:
            print(f"[FACE DETECTOR] ✗ Không tìm thấy video .avi trong: {videos_dir}")
            return
        
        print(f"[FACE DETECTOR] ✓ Tìm thấy {len(avi_files)} video(s):")
        for i, video in enumerate(avi_files, 1):
            print(f"  {i}. {video.name}")
        
        # Xử lý từng video
        for i, video_path in enumerate(avi_files, 1):
            print(f"\n[FACE DETECTOR] Đang xử lý video {i}/{len(avi_files)}: {video_path.name}")
            self.run_on_video(str(video_path))
            print(f"[FACE DETECTOR] ✓ Hoàn thành video {i}/{len(avi_files)}\n")


def main():
    """Hàm chính"""
    import sys
    
    # Tạo detector
    try:
        detector = FaceDetector(
            model_path="models/yolov8n-face-lindevs.pt",
            conf_threshold=0.5
        )
    except Exception as e:
        print(f"Lỗi khởi tạo detector: {e}")
        return 1
    
    # Xử lý các tham số
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Xử lý tất cả videos trong data/videos/
            print("[FACE DETECTOR] Chế độ: Xử lý tất cả videos trong data/videos/")
            detector.process_all_videos_in_data()
        else:
            # Chạy trên video được chỉ định
            video_path = sys.argv[1]
            detector.run_on_video(video_path)
    else:
        # Chạy trên camera mặc định
        detector.run_on_camera()
    
    return 0


if __name__ == "__main__":
    exit(main())
