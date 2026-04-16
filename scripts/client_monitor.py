import socket
import cv2
import struct
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="Dia chi IP cua Raspberry Pi")
    args = ap.parse_args()
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("[CLIENT] Dang ket noi toi Robot {args.ip}:5555...")
    client_socket.connect((args.ip, 5555))
    print("[CLIENT] Ket noi thanh cong!")
    
    data = b""
    payload_size = struct.calcsize(">L") # 4 bytes
    
    print("\n" + "="*50)
    print(" HUONG DAN: Nhan phim 'G' de yeu cau Robot quay 30 giay.")
    print("            Nhan phim 'Q' de thoat Monitor.")
    print("="*50 + "\n")
    
    try:
        while True:
            # Nhận độ dài của khung hình (4 bytes)
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet: break
                data += packet
                
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            
            # Nhận đủ dữ liệu của một khung hình JPEG
            # Nhận độ dài của khung hình (4 bytes)
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet: 
                    # NẾU KHÔNG CÓ GÓI TIN, BÁO LỖI VÀ THOÁT CHỨ KHÔNG ĐỂ VĂNG UNPACK ERROR
                    print("\n[CLIENT] ROBOT DA NGAT KET NOI HOAC DUNG TRUYEN DATA!")
                    return 
                data += packet
                
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # Giải mã và hiển thị
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Vẽ giao diện Hướng dẫn
            cv2.putText(frame, "MONITOR MODE - Nhin qua mat Robot", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Nhan [G] de ACTION!", (10, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
            cv2.imshow("Director Monitor", frame)
            
            # Bắt sự kiện phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('g') or key == ord('G'):
                print("[CLIENT] DA BAM G! Dang gui lenh quay video xuong Robot...")
                client_socket.sendall(b'G')
                print("[CLIENT] Robot dang ghi hinh (Ban co the thay hinh bi dung 30s).")
            elif key == ord('q'):
                break
                
    except Exception as e:
        print("[CLIENT] Loi mang: {}".format(e))
    finally:
        cv2.destroyAllWindows()
        client_socket.close()

if __name__ == "__main__":
    main()