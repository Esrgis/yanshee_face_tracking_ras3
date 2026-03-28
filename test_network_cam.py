import cv2
import json

# Đọc IP từ file config
with open("config.json", 'r') as f:
    config = json.load(f)

url = config["testing_env"]["ip_camera_url"]
print(f"[*] Đang cố gắng kết nối tới não Yanshee tại: {url}")

# OpenCV bắt thẳng luồng mạng!
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("[!] THẤT BẠI: Không thể mở luồng mạng. Kiểm tra lại IP hoặc Tường lửa (Firewall) của Windows!")
    exit()

print("[+] KẾT NỐI THÀNH CÔNG! Đang truyền hình trực tiếp...")
print("    Nhấn phím 'q' để tắt cửa sổ.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Mất tín hiệu mạng từ Robot!")
        break
        
    # Tạo Window hiển thị trên Laptop
    cv2.imshow("Yanshee Live Stream - Laptop Window", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()