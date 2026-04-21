# client_monitor.py
import socket
import cv2
import struct
import numpy as np
import argparse
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="Dia chi IP cua Raspberry Pi")
    args = ap.parse_args()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("[CLIENT] Dang ket noi toi {}:5555...".format(args.ip))
    client_socket.connect((args.ip, 5555))
    print("[CLIENT] Ket noi thanh cong!")

    data = b""
    payload_size = struct.calcsize(">L")

    is_counting_down = False
    auto_quit_time   = 0

    print("\n" + "="*50)
    print(" HUONG DAN: Nhan phim 'G' de quay 30 giay.")
    print("            He thong tu dong thoat sau khi xong.")
    print("            Nhan phim 'Q' de thoat thu cong.")
    print("="*50 + "\n")

    try:
        while True:
            if is_counting_down and time.time() >= auto_quit_time:
                print("[CLIENT] Het thoi gian quay. Dang dong...")
                break

            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    print("[CLIENT] ROBOT DA NGAT KET NOI! HEADER")
                    return
                data += packet

            packed_msg_size = data[:payload_size]
            data            = data[payload_size:]
            msg_size        = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet:
                    print("[CLIENT] ROBOT DA NGAT KET NOI! DATA")
                    return
                data += packet

            frame_data = data[:msg_size]
            data       = data[msg_size:]

            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            frame   = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is None:
                print("[CLIENT] Frame loi - reset buffer...")
                data = b""
                continue

            if is_counting_down:
                time_left = int(auto_quit_time - time.time())
                cv2.putText(frame,
                    "RECORDING... AUTO QUIT IN {}s".format(time_left),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame,
                    "DIRECTOR MONITOR - Press G to Record",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Director Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if (key == ord('g') or key == ord('G')) and not is_counting_down:
                cv2.destroyAllWindows()
                clip_name = input("Ten clip (slow/normal/fast/scale): ").strip()
                if clip_name not in ["slow", "normal", "fast", "scale"]:
                    print("[CLIENT] Ten clip khong hop le, thu lai.")
                    continue
                print("[CLIENT] Dang gui lenh quay: {}...".format(clip_name))
                client_socket.sendall(("G:" + clip_name).encode())
                is_counting_down = True
                auto_quit_time   = time.time() + 32

            elif key == ord('q') or key == ord('Q'):
                break

    except Exception as e:
        print("[CLIENT] Loi mang: {}".format(e))
    finally:
        cv2.destroyAllWindows()
        client_socket.close()
        print("[CLIENT] Da don dep va thoat an toan.")

if __name__ == "__main__":
    main()