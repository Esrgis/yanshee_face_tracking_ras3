#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import cv2
import struct
import time
import select
import os

# Giữ nguyên độ phân giải thấp để xem preview mượt như cam_stream
W_STREAM, H_STREAM = 320, 240
# Độ phân giải cao để lưu video Ground Truth
W_REC, H_REC = 640, 480
FPS = 20

def record_ground_truth(cap, duration=30):
    print("\n[SERVER] >>> ACTION! DANG QUAY VIDEO {} GIAY...".format(duration))
    
    # Ép thông số bằng số ID (3=Width, 4=Height) để tránh lỗi Attribute
    cap.set(3, W_REC)
    cap.set(4, H_REC)
    
    if not os.path.isdir("../data/videos"):
        os.makedirs("../data/videos")
        
    out_path = "../data/videos/record_{}.avi".format(int(time.time()))
    
    # Khởi tạo chuẩn nén MJPG - Tương thích tốt với RPi3
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W_REC, H_REC))
    
    t0 = time.time()
    while (time.time() - t0) < duration:
        ret, frame = cap.read()
        if ret:
            writer.write(frame)
            # Không gửi stream trong khi quay để dành 100% tài nguyên ghi file
            
    writer.release()
    print("[SERVER] >>> STOP! DA LUU: {}".format(out_path))
    
    # Trả lại cấu hình giống cam_stream để tiếp tục preview
    cap.set(3, W_STREAM)
    cap.set(4, H_STREAM)

def main():
    # Khởi tạo Socket Server tại cổng 5555
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 5555))
    server_socket.listen(1)
    
    print("[SERVER] Dang cho Laptop ket noi tai cong 5555...")
    conn, addr = server_socket.accept()
    print("[SERVER] Laptop {} ket noi thanh cong!".format(addr))
    
    # Khởi tạo Camera giống hệt cam_stream.py
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Robot khong mo duoc Camera!")
        return

    # Cấu hình y hệt cam_stream.py
    cap.set(3, W_STREAM)
    cap.set(4, H_STREAM)
    
    try:
        while True:
            # 1. Kiểm tra lệnh 'G' từ Laptop (Director)
            ready_to_read, _, _ = select.select([conn], [], [], 0.01)
            if ready_to_read:
                data = conn.recv(1024)
                if b'G' in data or b'g' in data:
                    record_ground_truth(cap, duration=30)
            
            # 2. Đọc frame và gửi stream (Tương đương logic generate của Flask)
            ret, frame = cap.read()
            if not ret: continue
            
            # Nén JPEG (Dùng ID 1 thay cho IMWRITE_JPEG_QUALITY)
            _, buffer = cv2.imencode('.jpg', frame, [1, 70])
            
            # Đóng gói dữ liệu gửi qua Socket
            data_bytes = buffer.tobytes()
            msg = struct.pack(">L", len(data_bytes)) + data_bytes
            conn.sendall(msg)
            
    except Exception as e:
        print("\n[SERVER] Ket noi bi ngat: ", e)
    finally:
        cap.release()
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    main()