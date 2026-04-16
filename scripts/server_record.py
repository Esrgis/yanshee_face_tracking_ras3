    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    import socket
    import cv2
    import struct
    import time
    import select
    import os

    W_STREAM, H_STREAM = 320, 240
    W_REC, H_REC = 640, 480
    FPS = 20

    def record_ground_truth(cap, duration=30):
        print("\n[SERVER] BAT DAU QUAY VIDEO {} GIAY...".format(duration))
        # Dung so ID 3 va 4 de tuong thich OpenCV cu
        cap.set(3, W_REC)
        cap.set(4, H_REC)
        
        if not os.path.isdir("../data/videos"):
            os.makedirs("../data/videos")
            
        out_path = "../data/videos/record_{}.avi".format(int(time.time()))
        
        # Cach khoi tao FourCC an toan cho ca CV2 va CV3
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        except AttributeError:
            fourcc = cv2.cv.CV_FOURCC(*'MJPG')

        writer = cv2.VideoWriter(out_path, fourcc, FPS, (W_REC, H_REC))
        
        t0 = time.time()
        while (time.time() - t0) < duration:
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
                
        writer.release()
        print("[SERVER] DA LUU FILE VAO: {}".format(out_path))
        
        # Trả độ phân giải về mức thấp để stream tiếp
        cap.set(3, W_STREAM)
        cap.set(4, H_STREAM)

    def main():
        # Khởi tạo Socket Server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 5555))
        server_socket.listen(1)
        
        print("[SERVER] Dang cho Laptop ket noi tai cong 5555...")
        conn, addr = server_socket.accept()
        print("[SERVER] Laptop {} da ket noi!".format(addr))
        
        cap = cv2.VideoCapture(0)
        
        # --- ĐOẠN KIỂM TRA CAMERA MỚI THÊM VÀO ---
        if not cap.isOpened():
            print("\n[SERVER ERROR] KHONG THE MO CAMERA!")
            print("Hay kiem tra xem co file nao khac dang chiem camera (Device Busy) khong.")
            conn.close()
            server_socket.close()
            return
        # ----------------------------------------
            
        # Dung so ID truc tiep de tranh loi AttributeError
        cap.set(3, W_STREAM) # 3 = CAP_PROP_FRAME_WIDTH
        cap.set(4, H_STREAM) # 4 = CAP_PROP_FRAME_HEIGHT
        
        try:
            while True:
                # 1. Kiểm tra xem có lệnh từ Laptop gửi sang không (Không chặn luồng)
                ready_to_read, _, _ = select.select([conn], [], [], 0.0)
                if ready_to_read:
                    data = conn.recv(1024)
                    if b'G' in data:
                        record_ground_truth(cap, duration=30)
                
                # 2. Đọc và truyền luồng video (Stream)
                ret, frame = cap.read()
                if not ret: continue
                
                # Nén thành JPEG chất lượng 70% để tiết kiệm Wi-Fi
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                
                # Đóng gói kích thước ảnh (chuẩn 4 bytes) + Dữ liệu ảnh
                msg = struct.pack(">L", len(buffer)) + buffer.tobytes()
                conn.sendall(msg)
                
        except Exception as e:
            print("\n[SERVER] Ngat ket noi: ", e)
        finally:
            cap.release()
            conn.close()
            server_socket.close()

    if __name__ == "__main__":
        main()