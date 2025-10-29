\
#!/usr/bin/env python3
# app.py — Flask + OpenCV MJPEG stream with improved tracking + strong blur and recording
from flask import Flask, render_template, Response, jsonify, request
import threading, time, cv2, datetime, numpy as np

app = Flask(__name__)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Minimal track class (shared with camera)
class Track:
    def __init__(self, bbox, tid, alpha=0.6):
        self.bbox = bbox
        self.id = tid
        self.missed = 0
        self.hits = 1
        self.alpha = alpha

    def update(self, bbox):
        x,y,w,h = bbox
        px,py,pw,ph = self.bbox
        nx = int(self.alpha * x + (1-self.alpha) * px)
        ny = int(self.alpha * y + (1-self.alpha) * py)
        nw = int(self.alpha * w + (1-self.alpha) * pw)
        nh = int(self.alpha * h + (1-self.alpha) * ph)
        self.bbox = (nx, ny, nw, nh)
        self.missed = 0
        self.hits += 1

    def mark_missed(self):
        self.missed += 1

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def strong_blur_roi(frame, roi_coords, strength=12):
    x1,y1,x2,y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    pixel_factor = max(2, int(min(w, h) / max(2, strength)))
    small_w = max(1, w // pixel_factor)
    small_h = max(1, h // pixel_factor)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixel = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(1, (pixel_factor // 2) | 1)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(pixel, (k, k), 0)
    frame[y1:y2, x1:x2] = blurred

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera (index {}).".format(src))
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.lock = threading.Lock()
        self.frame = None
        self.recording = False
        self.writer = None
        self.running = True
        self.tracks = []
        self.next_id = 0
        self.max_missed = 40
        self.blur_strength = 12
        t = threading.Thread(target=self._update_loop, daemon=True)
        t.start()

    def _update_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self._process(frame)

    def _process(self, frame):
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30,30))
        matched = set()
        used_det = set()
        for di, det in enumerate(detections):
            best_iou = 0.0
            best_t = None
            for ti, tr in enumerate(self.tracks):
                cur_iou = iou(det, tr.bbox)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_t = ti
            if best_iou > 0.25 and best_t is not None:
                self.tracks[best_t].update(det)
                matched.add(best_t)
                used_det.add(di)
        for di, det in enumerate(detections):
            if di in used_det:
                continue
            self.tracks.append(Track(tuple(det), self.next_id))
            self.next_id += 1
        updated_ids = set()
        for ti in matched:
            if ti < len(self.tracks):
                updated_ids.add(self.tracks[ti].id)
        for tr in self.tracks[:]:
            if tr.id not in updated_ids:
                tr.mark_missed()
            if tr.missed > self.max_missed:
                self.tracks.remove(tr)
        # apply blur to all tracks (even if temporarily missed)
        for tr in self.tracks:
            x,y,w,h = tr.bbox
            ex = int(w * 0.5)
            ey = int(h * 0.6)
            x1 = max(0, x - ex)
            y1 = max(0, y - ey)
            x2 = min(w_frame, x + w + ex)
            y2 = min(h_frame, y + h + ey)
            try:
                strong_blur_roi(frame, (x1,y1,x2,y2), strength=self.blur_strength)
            except Exception:
                pass
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(frame, f"ID:{tr.id}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        if self.recording and self.writer is not None:
            self.writer.write(frame)
            cv2.putText(frame, "REC", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        self.frame = frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, jpg = cv2.imencode('.jpg', self.frame)
            return jpg.tobytes()

    def start_recording(self, filename=None):
        with self.lock:
            if self.recording:
                return False
            if filename is None:
                filename = datetime.datetime.now().strftime("output_%Y%m%d_%H%M%S.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.writer = cv2.VideoWriter(filename, fourcc, fps, (w,h))
            self.recording = True
            self.current_filename = filename
            return True

    def stop_recording(self):
        with self.lock:
            if not self.recording:
                return False
            self.recording = False
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            return True

    def set_blur_strength(self, val):
        with self.lock:
            try:
                v = int(val)
                self.blur_strength = max(2, min(40, v))
            except:
                pass

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        if self.cap.isOpened():
            self.cap.release()

camera = None

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    global camera
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_record', methods=['POST'])
def start_record():
    global camera
    ok = camera.start_recording()
    return jsonify({"recording": ok, "filename": getattr(camera, 'current_filename', None)})

@app.route('/stop_record', methods=['POST'])
def stop_record():
    global camera
    ok = camera.stop_recording()
    return jsonify({"recording": not ok})

@app.route('/status')
def status():
    return jsonify({"recording": camera.recording, "blur_strength": camera.blur_strength})

@app.route('/set_blur', methods=['POST'])
def set_blur():
    data = request.get_json() or {}
    val = data.get('value', None)
    camera.set_blur_strength(val)
    return jsonify({"ok": True, "blur_strength": camera.blur_strength})

if __name__ == '__main__':
    try:
        camera = Camera(0)
    except Exception as e:
        print("ERROR starting camera:", e)
        exit(1)
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        if camera:
            camera.shutdown()
