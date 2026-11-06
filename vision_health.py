"""
vision_health.py - Nhan dien dau hieu bat thuong tren khuon mat bang YOLO (tuy chon).

QUAN TRONG:
- Model YOLO co san (COCO) KHONG biet "chan thuong", "mat nhot nhat"...
  Ban can tu huan luyen (hoac dung) mot model YOLO co cac lop tuong ung,
  dat tai  models/health_yolo.pt  roi khai bao ten lop trong CLASS_MAP.
- Neu khong co ultralytics hoac khong co file model: module tu tat (enabled=False),
  he thong phat hien buon ngu van chay binh thuong.
- Chi tra ve "dau hieu quan sat duoc", KHONG chan doan benh.
- Chay thread rieng, bo qua frame khi dang ban => khong lam cham detection_loop.
"""
import os
import time
import threading
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEALTH_MODEL_PATH = os.path.join(BASE_DIR, "models", "health_yolo.pt")

# ten lop trong model cua ban -> nhan hien thi tieng Viet
CLASS_MAP = {
    "injury": "Dấu hiệu chấn thương trên mặt",
    "blood": "Dấu hiệu chảy máu",
    "bruise": "Vết bầm trên mặt",
    "pale": "Sắc mặt nhợt nhạt",
    "distress": "Biểu hiện đau đớn / bất thường",
}

CONF_THRESHOLD = 0.55     # do tin cay toi thieu cua moi lan phat hien
PERSIST_SECONDS = 3.0     # dau hieu phai lien tuc >= 3 s moi tinh (chong bao gia)
FORGET_SECONDS = 1.5      # mat dau hieu > 1.5 s thi reset bo dem
INFER_INTERVAL = 0.4      # toi da ~2.5 lan suy luan / giay
IMG_SIZE = 320

log = logging.getLogger("drowsiness")


class HealthVision:
    def __init__(self, model_path=HEALTH_MODEL_PATH, class_map=None):
        self.class_map = dict(class_map or CLASS_MAP)
        self.model = None
        self.enabled = False
        self.reason_disabled = ""
        self._lock = threading.Lock()
        self._busy = False
        self._last_infer = 0.0
        self._first_seen = {}      # label -> ts lan dau thay lien tuc
        self._last_seen = {}       # label -> ts lan cuoi thay
        self._confirmed = {}       # label -> conf (da vuot PERSIST_SECONDS)

        if not os.path.exists(model_path):
            self.reason_disabled = "Chua co file model: " + model_path
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.enabled = True
        except Exception as exc:  # thieu thu vien / model hong
            self.reason_disabled = f"{type(exc).__name__}: {exc}"
            log.warning("HealthVision tat: %s", self.reason_disabled)

    # ------------------------------------------------------------
    def submit(self, bgr_frame):
        """Goi moi frame; tu bo qua neu dang ban / chua den luot. Khong chan."""
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            if self._busy or (now - self._last_infer) < INFER_INTERVAL:
                return
            self._busy = True
            self._last_infer = now
        frame = bgr_frame.copy()
        threading.Thread(target=self._infer, args=(frame,), daemon=True).start()

    def _infer(self, frame):
        try:
            results = self.model.predict(
                frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False
            )
            names = self.model.names
            found = {}
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_name = str(names[int(box.cls[0])])
                    if cls_name in self.class_map:
                        conf = float(box.conf[0])
                        found[cls_name] = max(conf, found.get(cls_name, 0.0))
            self._update(found, time.time())
        except Exception as exc:
            log.warning("HealthVision infer loi: %s", exc)
        finally:
            with self._lock:
                self._busy = False

    def _update(self, found, now):
        """Logic thoi gian (tach rieng de de test)."""
        with self._lock:
            for name, conf in found.items():
                self._last_seen[name] = now
                self._first_seen.setdefault(name, now)
                if now - self._first_seen[name] >= PERSIST_SECONDS:
                    self._confirmed[name] = conf
            for name in list(self._first_seen):
                if name not in found and now - self._last_seen.get(name, 0) > FORGET_SECONDS:
                    self._first_seen.pop(name, None)
                    self._confirmed.pop(name, None)

    def active_signs(self):
        """Danh sach dau hieu da duoc xac nhan (nhan tieng Viet)."""
        with self._lock:
            return [self.class_map[n] for n in self._confirmed if n in self.class_map]

    def reset(self):
        with self._lock:
            self._first_seen.clear()
            self._last_seen.clear()
            self._confirmed.clear()