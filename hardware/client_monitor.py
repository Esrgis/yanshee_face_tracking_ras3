import socket
import cv2
import struct
import numpy as np
import argparse
import time  # <--- THÊM THƯ VIỆN NÀY

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="Dia chi IP cua Raspberry Pi")
    args = ap.parse_args()
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("[CLIENT] Dang ket noi toi Robot {args.ip}:5555...")
    client_socket.connect((args.ip, 5555))
    print("[CLIENT] Ket noi thanh cong!")
    
    data = b""
    payload_size = struct.calcsize(">L")
    
    # --- BIẾN QUẢN LÝ THỜI GIAN TỰ HỦY ---
    is_counting_down = False
    auto_quit_time = 0
    
    print("\n" + "="*50)
    print(" HUONG DAN: Nhan phim 'G' de yeu cau Robot quay 30 giay.")
    print("            He thong se tu dong thoat sau khi hoan thanh.")
    print("            Nhan phim 'Q' de thoat Monitor thu cong.")
    print("="*50 + "\n")
    
    try:
        while True:
            # --- LOGIC TỰ ĐỘNG THOÁT ---
            if is_counting_down and time.time() >= auto_quit_time:
                print("\n[CLIENT] Đã hết thời gian quay video! Đang tự động đóng luồng...")
                break # Phá vỡ vòng lặp, chạy xuống khối finally để đóng socket
                
            # ==========================================
            # BƯỚC 1: NHẬN HEADER (Chỉ nhận đúng 4 bytes)
            # ==========================================
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet: 
                    print("\n[CLIENT] ROBOT DA NGAT KET NOI! HEADER")
                    return 
                data += packet
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            
            # ==========================================
            # BƯỚC 2: NHẬN DATA (Nhận đủ số bytes của ảnh)
            # ==========================================
            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet: 
                    print("\n[CLIENT] ROBOT DA NGAT KET NOI! DATA ")
                    return
                data += packet
            
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # ==========================================
            # BƯỚC 3: GIẢI MÃ VÀ HIỂN THỊ
            # ==========================================
            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("[CLIENT] Frame loi - Dang reset buffer...")
                data = b""
                continue

            # --- CẬP NHẬT GIAO DIỆN HIỂN THỊ ---
            if is_counting_down:
                # Tính thời gian còn lại
                time_left = int(auto_quit_time - time.time())
                cv2.putText(frame, "RECORDING... AUTO QUIT IN {time_left}s", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "DIRECTOR MONITOR - Press 'G' to Record", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
            cv2.imshow("Director Monitor", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if (key == ord('g') or key == ord('G')) and not is_counting_down:
                print("\n[CLIENT] DA BAM G! Dang gui lenh quay video...")
                client_socket.sendall(b'G')
                is_counting_down = True
                # Set thời gian tự hủy là 32 giây (30s cho video + 2s dự phòng độ trễ mạng)
                auto_quit_time = time.time() + 32 
                
            elif key == ord('q') or key == ord('Q'):
                break

    except Exception as e:
        print("[CLIENT] Loi mang: {e}")
    finally:
        cv2.destroyAllWindows()
        client_socket.close()
        print("[CLIENT] Da don dep tai nguyen va thoat chuong trinh an toan.")

if __name__ == "__main__":
    main()