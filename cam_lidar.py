#!/usr/bin/env python3
"""
Combined Camera (nhan dien vat the bang chip AI imx500 + nhan dien khuon mat bang OpenCV)
+ RPLIDAR web viewer cho Raspberry Pi.
Chay: python3 combined_stream.py
Xem:  http://<IP_CUA_PI>:8000/
"""

import io
import json
import logging
import os
import socketserver
import time
from http import server
from threading import Condition, Lock, Thread

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from rplidar import RPLidar

# ---------- Cau hinh ----------
LIDAR_PORT = '/dev/ttyUSB0'
LIDAR_BAUDRATE = 115200
CAM_SIZE = (640, 480)
MAX_DIST_MM = 6000
HTTP_PORT = 8000
JPEG_QUALITY = 80

# Model nhan dien vat the chinh thuc cua Raspberry Pi, cai qua "sudo apt install imx500-all"
MODEL_PATH = '/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk'
OBJ_THRESHOLD = 0.55  # do tin cay toi thieu de ve khung vat the

logging.basicConfig(level=logging.INFO)

# ---------- HTML trang chinh ----------
PAGE = """\
<html>
<head>
<title>Pi Camera (AI Object + Face) + LIDAR</title>
<style>
  body { background:#111; color:#eee; font-family:sans-serif; text-align:center; }
  .wrap { display:flex; justify-content:center; gap:30px; flex-wrap:wrap; margin-top:20px; }
  canvas, img { background:#000; border:1px solid #444; }
</style>
</head>
<body>
<h2>Raspberry Pi - Camera (Vat the qua AI Camera + Khuon mat) + LIDAR</h2>
<div class="wrap">
  <div>
    <h3>Camera</h3>
    <img src="stream.mjpg" width="640" height="480" />
  </div>
  <div>
    <h3>LIDAR (radar view)</h3>
    <canvas id="radar" width="500" height="500"></canvas>
  </div>
</div>
<script>
const canvas = document.getElementById('radar');
const ctx = canvas.getContext('2d');
const CX = 250, CY = 250, MAXR = 240;
const MAXDIST = %(maxdist)s;

function drawGrid() {
  ctx.strokeStyle = '#333';
  ctx.beginPath();
  for (let r = 1; r <= 4; r++) {
    ctx.moveTo(CX + r * (MAXR/4), CY);
    ctx.arc(CX, CY, r * (MAXR/4), 0, 2 * Math.PI);
  }
  ctx.moveTo(CX - MAXR, CY); ctx.lineTo(CX + MAXR, CY);
  ctx.moveTo(CX, CY - MAXR); ctx.lineTo(CX, CY + MAXR);
  ctx.stroke();
}

async function update() {
  try {
    const res = await fetch('/lidar_data');
    const data = await res.json();
    ctx.clearRect(0, 0, 500, 500);
    drawGrid();
    ctx.fillStyle = '#0f0';
    data.forEach(function (p) {
      const angleDeg = p[1];
      const dist = Math.min(p[2], MAXDIST);
      const angleRad = angleDeg * Math.PI / 180;
      const scale = MAXR / MAXDIST;
      const x = CX + dist * scale * Math.sin(angleRad);
      const y = CY - dist * scale * Math.cos(angleRad);
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  } catch (e) {
    console.error(e);
  }
  setTimeout(update, 200);
}
update();
</script>
</body>
</html>
""" % {"maxdist": MAX_DIST_MM}

# ---------- Bo nho dung frame moi nhat de stream ----------
class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = Condition()


output = StreamingOutput()


def find_haarcascade():
    """Tim file haarcascade_frontalface_default.xml o nhieu vi tri co the co
    (ban apt python3-opencv thuong khong co thuoc tinh cv2.data)."""
    candidates = []
    try:
        candidates.append(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except AttributeError:
        pass
    candidates += [
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ---------- Camera: object detection tren chip AI (imx500) + face detection (OpenCV CPU) ----------
def camera_worker():
    # IMX500 phai duoc khoi tao TRUOC khi tao Picamera2
    imx500 = IMX500(MODEL_PATH)
    intrinsics = getattr(imx500, 'network_intrinsics', None)
    labels = getattr(intrinsics, 'labels', None) if intrinsics is not None else None

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_video_configuration(
        main={"size": CAM_SIZE, "format": "RGB888"}, buffer_count=12
    )
    try:
        imx500.show_network_fw_progress_bar()
    except Exception:
        pass
    picam2.start(config)

    cascade_path = find_haarcascade()
    face_cascade = None
    if cascade_path:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            face_cascade = None
    else:
        logging.error('Khong tim thay haarcascade_frontalface_default.xml - bo qua nhan dien khuon mat')

    logging.info(
        'Camera san sang - AI object detection (imx500): ON, face detection (OpenCV): %s',
        bool(face_cascade)
    )

    debug_logged = False

    # --- Bien theo doi FPS ---
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    while True:
        request = None
        try:
            request = picam2.capture_request()
            metadata = request.get_metadata()
            frame = request.make_array("main")  # BGR order

            # --- Nhan dien vat the ngay tren chip AI cua camera (khong ton CPU Pi) ---
            try:
                outputs = imx500.get_outputs(metadata)
            except Exception:
                outputs = None

            if outputs is not None:
                try:
                    boxes, scores, classes = outputs[0], outputs[1], outputs[2]

                    if not debug_logged:
                        logging.info(
                            'DEBUG output shapes -> boxes:%s scores:%s classes:%s',
                            np.asarray(boxes).shape,
                            np.asarray(scores).shape,
                            np.asarray(classes).shape,
                        )
                        if len(np.asarray(scores)) > 0:
                            logging.info(
                                'DEBUG mau du lieu -> box[0]=%s score[0]=%s class[0]=%s',
                                np.asarray(boxes)[0] if np.asarray(boxes).ndim > 0 else boxes,
                                np.asarray(scores)[0] if np.asarray(scores).ndim > 0 else scores,
                                np.asarray(classes)[0] if np.asarray(classes).ndim > 0 else classes,
                            )
                        debug_logged = True

                    for box, score, cls in zip(boxes, scores, classes):
                        if score < OBJ_THRESHOLD:
                            continue
                        obj = imx500.convert_inference_coords(box, metadata, picam2)
                        x, y, w, h = obj.x, obj.y, obj.width, obj.height
                        cls_idx = int(cls)
                        name = labels[cls_idx] if labels and cls_idx < len(labels) else f"class{cls_idx}"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(
                            frame, f"{name} {score:.2f}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                        )
                except Exception as e:
                    if not debug_logged:
                        logging.warning('Loi parse object detection: %s', e)
                        debug_logged = True

            # --- Nhan dien khuon mat bang OpenCV (CPU, vi imx500 khong co model nay) ---
            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces, reject_levels, level_weights = face_cascade.detectMultiScale3(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50),
                    outputRejectLevels=True
                )
                for (x, y, w, h), weight in zip(faces, level_weights):
                    # Chuan hoa weight (thuong 0-15+) thanh % gan dung, gioi han 1-99%
                    confidence_pct = min(99, max(1, int(weight * 8)))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f'Face {confidence_pct}%', (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            # --- Tinh va ve FPS ---
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            ok, jpeg = cv2.imencode(
                '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                with output.condition:
                    output.frame = jpeg.tobytes()
                    output.condition.notify_all()
        except Exception as e:
            logging.warning('Loi xu ly camera: %s', e)
            time.sleep(1)
        finally:
            if request is not None:
                request.release()


# ---------- LIDAR background thread ----------
lidar_points = []
lidar_lock = Lock()


def lidar_worker():
    global lidar_points
    while True:
        lidar = None
        try:
            lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUDRATE)
            time.sleep(0.5)
            lidar.clean_input()
            for scan in lidar.iter_scans():
                pts = [[q, angle, dist] for (q, angle, dist) in scan]
                with lidar_lock:
                    lidar_points = pts
        except Exception as e:
            logging.warning('Loi LIDAR: %s - thu lai sau 5s', e)
            time.sleep(5)
        finally:
            if lidar is not None:
                try:
                    lidar.stop()
                    lidar.stop_motor()
                    lidar.disconnect()
                except Exception:
                    pass


# ---------- HTTP handler ----------
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    if frame is None:
                        continue
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Client camera ngat: %s', e)

        elif self.path == '/lidar_data':
            with lidar_lock:
                data = json.dumps(lidar_points).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    t_cam = Thread(target=camera_worker, daemon=True)
    t_cam.start()

    t_lidar = Thread(target=lidar_worker, daemon=True)
    t_lidar.start()

    try:
        srv = StreamingServer(('', HTTP_PORT), StreamingHandler)
        print(f"Server dang chay -> http://<IP_CUA_PI>:{HTTP_PORT}")
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Dang dung server...")
