# server_record.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import cv2
import struct
import time
import select
import os

W_STREAM, H_STREAM = 320, 240
FPS       = 20
MAX_FRAMES = 30 * FPS  # 600 frames = 30 giay

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

    is_recording = False
    writer       = None
    frame_count  = 0

    if not os.path.isdir("data/videos"):
        os.makedirs("data/videos")

    try:
        while True:
            # 1. Kiem tra lenh tu laptop
            ready_to_read, _, _ = select.select([conn], [], [], 0.01)
            if ready_to_read:
                raw = conn.recv(1024)
                decoded = raw.decode("utf-8", errors="ignore")
                if decoded.startswith("G:") and not is_recording:
                    clip_name = decoded.split(":")[1].strip()
                    if not clip_name:
                        clip_name = "unknown"
                    is_recording = True
                    frame_count  = 0
                    out_path = "data/videos/{}.avi".format(clip_name)
                    fourcc   = cv2.VideoWriter_fourcc(*'MJPG')
                    writer   = cv2.VideoWriter(out_path, fourcc, FPS, (W_STREAM, H_STREAM))
                    print("[SERVER] >>> ACTION! {} -> {}".format(clip_name, out_path))

            # 2. Doc frame
            ret, frame = cap.read()
            if not ret:
                continue

            # 3. Ghi frame
            if is_recording:
                if frame_count < MAX_FRAMES:
                    writer.write(frame)
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print("[SERVER] {} / {} frames".format(frame_count, MAX_FRAMES))
                else:
                    is_recording = False
                    writer.release()
                    writer = None
                    print("[SERVER] >>> STOP! Saved successfully.")

            # 4. Gui frame len monitor
            _, buffer   = cv2.imencode('.jpg', frame, [1, 50])
            data_bytes  = buffer.tobytes()
            msg         = struct.pack(">L", len(data_bytes)) + data_bytes
            conn.sendall(msg)

    except Exception as e:
        print("[SERVER] Loi: {}".format(e))
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        conn.close()
        server_socket.close()
        print("[SERVER] Da don dep tai nguyen.")

if __name__ == "__main__":
    main()