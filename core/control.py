class PIDController:
    def __init__(self, Kp=0.05, Ki=0.0, Kd=0.01, max_integral=50.0, deadzone=0.0, output_limit=100.0):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.max_integral = float(max_integral)
        self.deadzone = float(deadzone)
        self.output_limit = float(output_limit)
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        if dt <= 0:
            return 0.0
        if abs(error) < self.deadzone:
            error = 0.0
        P = self.Kp * error
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        I = self.Ki * self.integral
        D = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error
        out = P + I + D
        return max(-self.output_limit, min(self.output_limit, out))

    def reset_memory(self):
        self.integral = 0.0
        self.prev_error = 0.0