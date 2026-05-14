# -*- coding: utf-8 -*-
from abc import abstractmethod

class VisionSystem(object):
    # Interface cho tất cả vision backend
    @abstractmethod
    def process_frame(self, frame, prev_x=-1, prev_y=-1):
        # Input: frame, prev_x, prev_y
        # Output: (target_found, bbox, center_x, center_y)
        pass
