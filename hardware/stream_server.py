#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import select
import socket
import struct
import threading
import time

import cv2


class FrameStreamServer(object):
    def __init__(self, host="0.0.0.0", port=5555, fps=10, jpeg_quality=45,
                 record_dir="data/videos", record_fps=20, max_record_sec=30):
        self.host = host
        self.port = int(port)
        self.fps = max(1, int(fps))
        self.jpeg_quality = int(jpeg_quality)
        self.record_dir = record_dir
        self.record_fps = int(record_fps)
        self.max_record_frames = int(max_record_sec * record_fps)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_size = None
        self.running = False
        self.thread = None
        self.conn = None
        self.server_socket = None

        self.is_recording = False
        self.writer = None
        self.record_count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        print("[STREAM] Server ready on {}:{} | fps={} | q={}".format(
            self.host, self.port, self.fps, self.jpeg_quality))

    def stop(self):
        self.running = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except Exception:
                pass

    def update(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            self.latest_size = (frame.shape[1], frame.shape[0])

        if self.is_recording and self.writer is not None:
            self.writer.write(frame)
            self.record_count += 1
            if self.record_count >= self.max_record_frames:
                self._stop_recording()

    def _run(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)

        while self.running:
            if self.conn is None:
                try:
                    self.conn, addr = self.server_socket.accept()
                    self.conn.setblocking(False)
                    print("[STREAM] Client connected: {}".format(addr))
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print("[STREAM] Accept error: {}".format(e))
                    continue

            try:
                self._read_client_command()
                self._send_latest_frame()
                time.sleep(1.0 / self.fps)
            except Exception as e:
                print("[STREAM] Client disconnected: {}".format(e))
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None

    def _read_client_command(self):
        ready, _, _ = select.select([self.conn], [], [], 0)
        if not ready:
            return
        raw = self.conn.recv(1024)
        if not raw:
            raise RuntimeError("socket closed")
        decoded = raw.decode("utf-8", errors="ignore")
        if decoded.startswith("G:") and not self.is_recording:
            clip_name = decoded.split(":", 1)[1].strip() or "unknown"
            self._start_recording(clip_name)

    def _send_latest_frame(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            return

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        data = buffer.tobytes()
        self.conn.sendall(struct.pack(">L", len(data)) + data)

    def _start_recording(self, clip_name):
        if self.latest_size is None:
            return
        if not os.path.isdir(self.record_dir):
            os.makedirs(self.record_dir)
        out_path = os.path.join(self.record_dir, "{}.avi".format(clip_name))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(out_path, fourcc, self.record_fps, self.latest_size)
        self.record_count = 0
        self.is_recording = True
        print("[STREAM] Recording -> {}".format(out_path))

    def _stop_recording(self):
        self.is_recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        print("[STREAM] Recording saved")
