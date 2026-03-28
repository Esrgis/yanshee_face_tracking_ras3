# CHẠY FILE NÀY TRÊN BÊN TRONG CON ROBOT YANSHEE
import cv2
from flask import Flask, Response

app = Flask(__name__)
# Số 0 là cổng camera mặc định của Raspberry Pi
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def generate():
    while True:
        ret, frame = cap.read()
        if not ret: continue
        # Nén ảnh thành chuẩn JPEG cho nhẹ băng thông mạng
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Phát sóng trên cổng 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)