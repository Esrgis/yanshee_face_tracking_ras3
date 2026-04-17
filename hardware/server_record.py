#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/record_videos.py -- Quay 3 video benchmark tren robot/laptop

Chay:
  python scripts/record_videos.py --source 0 --duration 30
  python scripts/record_videos.py --source 0 --duration 30 --output data/videos

Output:
  data/videos/slow.avi
  data/videos/normal.avi
  data/videos/fast.avi

Huong dan:
  - slow  : nguoi di cham, quay mat nhe
  - normal: di binh thuong
  - fast  : di nhanh, xoay dau nhanh
  - scale : dung im, tien sat vao camera roi lui ra xa lien tuc
"""
import socket
import cv2
import struct
import time
import select
import os

W_STREAM, H_STREAM = 320, 240
FPS = 20

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 5555))
    server_socket.listen(1)
    
    print("[SERVER] Dang cho Client ket noi tai cong 5555...")
    conn, addr = server_socket.accept()
    print("[SERVER] Client {} ket noi thanh cong!".format(addr))
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc Camera!")
        return

    cap.set(3, W_STREAM)
    cap.set(4, H_STREAM)
    cap.set(38, 1) 
    
# --- CÁC BIẾN QUẢN LÝ TRẠNG THÁI QUAY VIDEO (STATE MACHINE) ---
    is_recording = False
    writer = None
    frame_count = 0
    MAX_FRAMES = 30 * FPS  # 30 giây * 20 fps = 600 frames
    
    if not os.path.isdir("../data/videos"):
        os.makedirs("../data/videos")

    try:
        while True:
            # 1. KIỂM TRA LỆNH TỪ LAPTOP
            ready_to_read, _, _ = select.select([conn], [], [], 0.01)
            if ready_to_read:
                data = conn.recv(1024)
                if (b'G' in data or b'g' in data) and not is_recording:
                    is_recording = True
                    frame_count = 0  # Reset bộ đếm frame
                    
                    out_path = "../data/videos/record_{}.avi".format(int(time.time()))
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W_STREAM, H_STREAM))
                    print("\n[SERVER] >>> ACTION! START RECORDING {MAX_FRAMES} FRAMES TO {out_path}...")
            
            # 2. ĐỌC FRAME TỪ CAMERA
            ret, frame = cap.read()
            if not ret: continue
            
            # 3. GHI FRAME XUỐNG DISK (Đếm theo frame, không đếm theo giây)
            if is_recording:
                if frame_count < MAX_FRAMES:
                    writer.write(frame)
                    frame_count += 1
                else:
                    # Ghi đủ 600 frame thì dọn dẹp và tắt cờ
                    is_recording = False
                    writer.release()
                    writer = None
                    print("[SERVER] >>> STOP! RECORDING SAVED SUCCESSFULLY.")
            
            # 4. GỬI FRAME LÊN MONITOR CHO ĐẠO DIỄN
            _, buffer = cv2.imencode('.jpg', frame, [1, 50])
            data_bytes = buffer.tobytes()
            msg = struct.pack(">L", len(data_bytes)) + data_bytes
            conn.sendall(msg)
            
    except Exception as e:
        print("\n[SERVER] Ket noi bi ngat hoac loi: ", e)
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    main()