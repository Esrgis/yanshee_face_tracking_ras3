from abc import ABC, abstractmethod

class VisionSystem(ABC):
    """
    Interface chung cho tất cả backend Vision.
    Không bao giờ sửa file này khi thêm backend mới.
    """
    @abstractmethod
    def process_frame(self, frame):
        """
        Đầu vào : frame BGR (numpy array)
        Đầu ra  : (target_found, bbox, center_x, center_y)
                   bool          tuple  int         int
        """
        pass