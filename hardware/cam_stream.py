# Run on Yanshee robot
import cv2
from flask import Flask, Response

app = Flask(__name__)
# Camera 0 is Raspberry Pi default
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def generate():
    while True:
        ret, frame = cap.read()
        if not ret: continue
        # Compress to JPEG for network
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Stream on port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)