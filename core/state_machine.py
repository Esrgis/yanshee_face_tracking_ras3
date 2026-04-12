import time
from enum import Enum

class RobotState(Enum):
    TRACKING = 1
    SATURATED = 2
    SEARCH = 3
    LOST = 4

class TrackingStateMachine:
    def __init__(self, timeout_lost=3.0, lost_frame_threshold=5):
        self.current_state = RobotState.SEARCH
        self.lost_start_time = time.time()
        self.timeout_lost = timeout_lost
        self.lost_counter = 0
        self.lost_frame_threshold = lost_frame_threshold

    def update(self, target_found, predicted_angle, max_angle, min_angle):
        if self.current_state == RobotState.TRACKING:
            if not target_found:
                self.current_state = RobotState.SEARCH
                self.lost_start_time = time.time()
                self.lost_counter = 0
            elif predicted_angle >= max_angle or predicted_angle <= min_angle:
                self.current_state = RobotState.SATURATED

        elif self.current_state == RobotState.SATURATED:
            self.current_state = RobotState.SEARCH
            self.lost_start_time = time.time()
            self.lost_counter = 0

        elif self.current_state == RobotState.SEARCH:
            if target_found:
                self.current_state = RobotState.TRACKING
                self.lost_counter = 0
            else:
                self.lost_counter += 1
                time_elapsed = time.time() - self.lost_start_time
                if time_elapsed > self.timeout_lost and self.lost_counter >= self.lost_frame_threshold:
                    self.current_state = RobotState.LOST

        elif self.current_state == RobotState.LOST:
            if target_found:
                self.current_state = RobotState.TRACKING
                self.lost_counter = 0

        return self.current_state

    def get_state(self):
        return self.current_state

    def get_state_name(self):
        return self.current_state.name