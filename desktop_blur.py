\
#!/usr/bin/env python3
# desktop_blur.py
# Лёгкий локальный скрипт: детектирование → трекинг (удержание) → усиленный блюр (пикселизация + gaussian)
import cv2
import numpy as np
from datetime import datetime
import time, os

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def iou(boxA, boxB):
    # box = (x,y,w,h)
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

class Track:
    def __init__(self, bbox, tid, alpha=0.6):
        self.bbox = bbox  # smoothed bbox (x,y,w,h)
        self.id = tid
        self.missed = 0
        self.hits = 1
        self.alpha = alpha  # EMA smoothing

    def update(self, bbox):
        # Exponential smoothing to avoid jitter
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

def strong_blur_roi(frame, roi_coords, strength=12):
    # roi_coords = (x1,y1,x2,y2)
    x1,y1,x2,y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    # pixelation factor depends on strength and face size
    pixel_factor = max(2, int(min(w, h) / max(2, strength)))
    small_w = max(1, w // pixel_factor)
    small_h = max(1, h // pixel_factor)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixel = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    # additional Gaussian to soften blocks
    k = max(1, (pixel_factor // 2) | 1)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(pixel, (k, k), 0)
    frame[y1:y2, x1:x2] = blurred

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Если используешь удалённую машину, убедись, что камера доступна как устройство 0.")
        return
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    tracks = []
    next_id = 0
    max_missed = 40  # количество кадров, в течение которых будем держать область без новых детекций
    blur_strength = 12  # можно менять клавишами ] и [

    recording = False
    out = None

    print("Запущено. Нажми 'r' — старт/стоп записи, ']' увеличить blur, '[' уменьшить blur, 'q' — выход")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # более агрессивные параметры детекции: пытаемся ловить частичные лица
        detections = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30,30))

        matched = set()
        used_det = set()

        # матчим детекции с треками по IoU
        for di, det in enumerate(detections):
            best_iou = 0.0
            best_t = None
            for ti, tr in enumerate(tracks):
                cur_iou = iou(det, tr.bbox)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_t = ti
            if best_iou > 0.25 and best_t is not None:
                tracks[best_t].update(det)
                matched.add(best_t)
                used_det.add(di)

        # создаём новые треки для неиспользованных детекций
        for di, det in enumerate(detections):
            if di in used_det:
                continue
            tracks.append(Track(tuple(det), next_id))
            next_id += 1

        # помечаем треки как пропущенные, удаляем старые
        for tr in tracks[:]:
            if tr not in [tracks[i] for i in matched if i < len(tracks)]:
                # if it wasn't updated this frame, increment missed
                # but above matching uses indices that might shift; instead use a safer approach:
                pass
        # Safer loop: compare track ids updated by matched set
        updated_ids = set()
        for ti in matched:
            if ti < len(tracks):
                updated_ids.add(tracks[ti].id)
        for tr in tracks[:]:
            if tr.id not in updated_ids:
                tr.mark_missed()
            if tr.missed > max_missed:
                tracks.remove(tr)

        # draw and blur all active tracks (even если детекция пропала — удерживаем)
        for tr in tracks:
            x,y,w,h = tr.bbox
            # расширяем область, чтобы покрыть руку или частичное лицо
            ex = int(w * 0.5)
            ey = int(h * 0.6)
            x1 = max(0, x - ex)
            y1 = max(0, y - ey)
            x2 = min(w_frame, x + w + ex)
            y2 = min(h_frame, y + h + ey)
            try:
                strong_blur_roi(frame, (x1,y1,x2,y2), strength=blur_strength)
            except Exception as e:
                pass
            # рисуем рамку и id для отладки
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(frame, f"ID:{tr.id}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if recording and out is not None:
            out.write(frame)
            cv2.putText(frame, "REC", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Face Blur Live - desktop (improved)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
            if recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"output_{ts}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(fname, fourcc, fps, (w,h))
                print("Started recording ->", fname)
            else:
                if out is not None:
                    out.release()
                    out = None
                print("Stopped recording.")
        elif key == ord('q'):
            break
        elif key == ord(']'):
            blur_strength = max(3, blur_strength - 1)  # уменьшение числа = сильнее пикселизации
            print("Blur strength:", blur_strength)
        elif key == ord('['):
            blur_strength = blur_strength + 1
            print("Blur strength:", blur_strength)

    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
