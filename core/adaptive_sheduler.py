#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adaptive_scheduler.py
---------------------
Velocity-aware Adaptive Detection Scheduler cho embedded face tracking.
 
Vấn đề cần giải quyết
----------------------
detection_skip cố định là một trade-off thủ công:
  - skip nhỏ (=1): detect mỗi frame → chính xác nhưng CPU nặng, FPS thấp
  - skip lớn (=10): nhẹ CPU nhưng mất target khi chuyển động nhanh
 
Không có giá trị skip nào "tối ưu" cho mọi tình huống —
vì "tình huống" thay đổi liên tục trong thực tế.
 
Ý tưởng chính
--------------
Thay vì chọn một con số cố định, scheduler quan sát hai tín hiệu
từ chính pipeline hiện tại (không cần sensor bổ sung):
 
  1. velocity   : tốc độ di chuyển của target (pixel/frame)
                  → target nhanh → cần detect thường hơn
  2. jitter     : độ dao động của bbox (Kalman residual, pixel)
                  → tracker đang bất ổn → cần confirm bằng detection
 
Công thức:
  skip(t) = clip( base_skip / (1 + α·vel + β·jit), min_skip, max_skip )
 
Với α, β là hyperparameter — được tune và báo cáo trong ablation study.
 
Ngoài ra scheduler nhận RobotState từ FSM để override cứng:
  - SEARCH / LOST   → skip=1 (detect liên tục để tìm lại target)
  - SATURATED       → skip=1 (servo đang giới hạn, cần relocalize nhanh)
  - TRACKING        → dùng công thức adaptive bình thường
 
Cách bật/tắt
------------
Trong config.json:
  "adaptive_scheduler": {
      "enabled": true,   ← false = dùng detection_skip cố định như cũ
      ...
  }
 
Khi enabled=false, class vẫn tồn tại nhưng luôn trả về base_skip.
Điều này cho phép A/B test sạch giữa static và adaptive
chỉ bằng cách đổi một flag trong config — không sửa code.
 
Paper reference
---------------
Công thức velocity-aware scheduling tương tự được dùng trong:
- "Adaptive frame-rate control for real-time object tracking"
  (Lim et al., Pattern Recognition Letters, 2018)
- "Resource-aware visual tracking" (Luber et al., IROS 2019)
Nhưng chưa có paper nào apply cụ thể cho humanoid robot + PID servo
với FSM-aware override — đó là novelty của chúng ta.
"""
 
import collections
import time
from core.state_machine import RobotState
 
 
class AdaptiveDetectionScheduler:
    """
    Tính detection_skip tối ưu theo thời gian thực.
 
    Parameters
    ----------
    enabled : bool
        True  → chạy adaptive logic
        False → luôn trả về base_skip (tương đương pipeline cũ)
 
    base_skip : int
        Điểm xuất phát — giá trị skip khi target đứng yên hoàn toàn.
        Tương đương detection_skip_frames trong config cũ.
 
    min_skip : int
        Giới hạn dưới — không detect dày hơn mức này dù target nhanh đến đâu.
        Thường = 1.
 
    max_skip : int
        Giới hạn trên — không skip nhiều hơn mức này dù target đứng yên.
        Giúp tránh tracker drift hoàn toàn mà không có detection confirm.
 
    alpha : float
        Weight cho velocity. Tăng → scheduler phản ứng mạnh hơn với chuyển động.
        Giá trị khởi đầu gợi ý: 0.08 (tune qua ablation).
 
    beta : float
        Weight cho jitter. Tăng → scheduler detect nhiều hơn khi tracker bất ổn.
        Giá trị khởi đầu gợi ý: 0.05 (tune qua ablation).
 
    velocity_window : int
        Số frame dùng để tính velocity trung bình — giảm noise từng frame.
        Thường = 5.
    """
 
    def __init__(
        self,
        enabled=True,
        base_skip=5,
        min_skip=1,
        max_skip=15,
        alpha=0.08,
        beta=0.05,
        velocity_window=5,
    ):
        self.enabled   = enabled
        self.base_skip = int(base_skip)
        self.min_skip  = int(min_skip)
        self.max_skip  = int(max_skip)
        self.alpha     = float(alpha)
        self.beta      = float(beta)
 
        # Buffer tọa độ để tính velocity trung bình
        self._cx_history = collections.deque(maxlen=velocity_window)
        self._cy_history = collections.deque(maxlen=velocity_window)
 
        # Skip hiện tại — để bên ngoài có thể đọc/log
        self.current_skip = base_skip
 
        # Telemetry — dùng để ghi vào CSV cho paper
        self.last_velocity = 0.0
        self.last_jitter   = 0.0
        self.last_reason   = "init"
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def compute_skip(self, cx, cy, jitter, robot_state):
        """
        Tính detection_skip cho frame tiếp theo.
 
        Parameters
        ----------
        cx, cy : int
            Tọa độ tâm của target frame này (từ Kalman filtered output).
            Truyền -1 nếu không có target.
 
        jitter : float
            Kalman residual = |raw_cx - filtered_cx| (pixel).
            Đo độ bất ổn của tracker. Truyền 0.0 nếu không có Kalman.
 
        robot_state : RobotState
            State hiện tại từ FSM — dùng để override cứng.
 
        Returns
        -------
        int
            detection_skip để dùng cho frame tiếp theo.
        """
 
        # --- Nếu disabled: trả về base_skip luôn, không tính gì ---
        if not self.enabled:
            self.current_skip = self.base_skip
            self.last_reason  = "disabled"
            return self.current_skip
 
        # --- Override cứng theo FSM state ---
        # Khi đang tìm kiếm hoặc mất target, detect liên tục
        if robot_state in (RobotState.SEARCH, RobotState.LOST, RobotState.SATURATED):
            self.current_skip = self.min_skip
            self.last_reason  = "fsm_override_{}".format(robot_state.name)
            self._update_history(cx, cy)  # vẫn update history để velocity sẵn sàng
            return self.current_skip
 
        # --- Tính velocity ---
        velocity = self._compute_velocity(cx, cy)
        self._update_history(cx, cy)
 
        # --- Công thức adaptive ---
        # skip giảm khi velocity hoặc jitter tăng
        # denominator > 1 → skip < base_skip
        denominator = 1.0 + self.alpha * velocity + self.beta * jitter
        raw_skip    = self.base_skip / denominator
        new_skip    = int(round(raw_skip))
        new_skip    = max(self.min_skip, min(self.max_skip, new_skip))
 
        # Ghi telemetry
        self.last_velocity = round(velocity, 2)
        self.last_jitter   = round(jitter, 2)
        self.last_reason   = "adaptive"
        self.current_skip  = new_skip
 
        return self.current_skip
 
    def reset(self):
        """Gọi khi target bị mất để xóa history velocity."""
        self._cx_history.clear()
        self._cy_history.clear()
        self.last_velocity = 0.0
        self.last_jitter   = 0.0
        self.last_reason   = "reset"
 
    def get_telemetry(self):
        """
        Trả về dict để ghi vào CSV log.
        Thêm các cột này vào csv_writer trong main_tracker_robot.py.
        """
        return {
            "sched_skip"    : self.current_skip,
            "sched_velocity": self.last_velocity,
            "sched_jitter"  : self.last_jitter,
            "sched_reason"  : self.last_reason,
        }
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _compute_velocity(self, cx, cy):
        """
        Tính velocity trung bình (pixel/frame) từ window lịch sử.
        Dùng trung bình thay vì frame-to-frame để giảm noise.
        """
        if cx < 0 or cy < 0:
            return 0.0
 
        if len(self._cx_history) < 2:
            return 0.0
 
        # Tính displacement giữa frame hiện tại và frame đầu window
        oldest_cx = self._cx_history[0]
        oldest_cy = self._cy_history[0]
        n         = len(self._cx_history)
 
        dx = (cx - oldest_cx) / n
        dy = (cy - oldest_cy) / n
 
        return (dx**2 + dy**2) ** 0.5  # Euclidean distance / frame
 
    def _update_history(self, cx, cy):
        if cx >= 0 and cy >= 0:
            self._cx_history.append(cx)
            self._cy_history.append(cy)
 
 
# ------------------------------------------------------------------
# Factory: tạo từ config.json
# ------------------------------------------------------------------
 
def build_scheduler_from_config(config):
    """
    Tạo AdaptiveDetectionScheduler từ dict config.json.
 
    Cách dùng trong main:
        scheduler = build_scheduler_from_config(config)
 
    Nếu key "adaptive_scheduler" không có trong config,
    trả về scheduler với enabled=False (safe fallback).
    """
    sched_cfg = config.get("adaptive_scheduler", {})
 
    return AdaptiveDetectionScheduler(
        enabled         = sched_cfg.get("enabled", False),
        base_skip       = sched_cfg.get("base_skip",
                            config.get("camera", {}).get("detection_skip_frames", 3)),
        min_skip        = sched_cfg.get("min_skip", 1),
        max_skip        = sched_cfg.get("max_skip", 15),
        alpha           = sched_cfg.get("alpha", 0.08),
        beta            = sched_cfg.get("beta", 0.05),
        velocity_window = sched_cfg.get("velocity_window", 5),
    )