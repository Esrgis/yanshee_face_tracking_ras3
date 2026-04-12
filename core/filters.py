import numpy as np
import cv2

class TrackerKalmanFilter:
    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)

    def init_state(self, x, y):
        self.kf.statePre = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
        self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)

    def predict(self):
        pred = self.kf.predict()
        return int(pred[0,0]), int(pred[1,0])

    def update(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)
        state = self.kf.statePost
        return int(state[0,0]), int(state[1,0])