"""
Driver Guard - Giam sat dau hieu buon ngu cua tai xe (v4)
---------------------------------------------------------
Thuat toan phat hien (EAR/MAR/gat gu/cui sau, thich ung moi truong, watchdog,
camera bi che, ghi log) giu nguyen tu v3. v4: giao dien moi (customtkinter),
luong nhac dung nghi (fatigue_advisor), nguon dia diem poi_provider.
Chay:  python main.py
"""

import settings  # nap .env truoc cac module dung API key
import cv2
import datetime
import time
import csv
import os
import logging
import threading
import traceback
import numpy as np
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont
import re
import customtkinter as ctk
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import winsound
from datetime import datetime
import route_map
import poi_provider
import ui_theme as T
from route_map import RoutePlanner
from fatigue_advisor import AdvisorService, AdvicePopup
from vision_health import HealthVision
# RAG la chuc nang doc lap; khong goi trong detection_loop
RAG_IMPORT_ERROR = ""
try:
    from rag_gemini import RAGService, DOCUMENTS_DIR
except Exception as _e:
    RAGService = None
    RAG_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
LOG_PATH = os.path.join(BASE_DIR, "alerts_log.csv")
ERROR_LOG_PATH = os.path.join(BASE_DIR, "system_errors.log")

# ====================== NGƯỠNG ======================
EAR_THRESHOLD_NORMAL = 0.18
EAR_THRESHOLD_STRICT = 0.13
EAR_THRESHOLD_YAW = 0.25
EAR_THRESHOLD_EXTREME = 0.28
EAR_VERY_LOW = 0.12
EAR_THRESHOLD_LOWLIGHT = 0.23

EYE_CLOSED_TIME_LIMIT = 1.8
MAR_THRESHOLD = 0.65
YAWN_TIME_LIMIT = 0.9
BEEP_INTERVAL = 0.2
FACE_LOST_GRACE = 0.7

PITCH_HIGH = 0.40
YAW_HIGH = 0.35
YAW_EXTREME = 0.65
SMILE_RATIO_THRESHOLD = 0.6

NOD_DROP_THRESHOLD = 0.45
NOD_RECOVER_THRESHOLD = 0.41
NOD_COUNT_NEEDED = 2
NOD_TIME_WINDOW = 6.5
NOD_HOLD_TIME = 1.5

DEEP_PITCH_THRESHOLD = 0.55
DEEP_HOLD_TIME = 2.2
DEEP_NO_FACE_GRACE = 3.5

LOW_LIGHT_MEAN = 55
GLARE_CLIP_RATIO = 0.12
GLARE_EYE_MEAN_MIN = 215.0
# Rung/chuyển động: KHÔNG khóa nod — chỉ siết thời gian giữ cúi/ngẩng
MOTION_DIFF_THRESH = 14.0
NOD_PITCH_SMOOTH = 0.35
NOD_MIN_GAP = 0.50
NOD_DWELL_NORMAL = 0.10     # phải giữ cúi/ngẩng tối thiểu (s)
NOD_DWELL_NOISY = 0.25      # khi frame nhiễu (rung): giữ lâu hơn mới đếm 1 gật

EAR_SMOOTH_ALPHA = 0.45
OPEN_RESET_FRAMES = 4

SUNGLASS_RATIO_NORMAL = 0.62
SUNGLASS_RATIO_LOWLIGHT = 0.48
SUNGLASS_CONFIRM_FRAMES = 5

CAM_FAIL_REOPEN = 30

# ----- An toàn / watchdog -----
HEARTBEAT_TIMEOUT = 2.5          # giây không có frame/detect → SYSTEM ERROR
WATCHDOG_CHECK_MS = 500
MAX_DETECT_RESTARTS = 2
OBSTRUCTED_HOLD_SEC = 2.0
# 1) Flatness trên ảnh blur (bắt che mọi màu)
# 2) Structure recovery bằng CLAHE (phòng tối có mặt → lộ cạnh → không obstructed)
OBSTRUCTED_LAP_VAR_MAX = 18.0
OBSTRUCTED_LOW_VAR_RATIO = 0.70
OBSTRUCTED_LOCAL_STD_MAX = 7.0
OBSTRUCTED_CENTER_STD_MAX = 8.0
OBSTRUCTED_CENTER_LAP_MAX = 14.0
OBSTRUCTED_LOCAL_KERNEL = 15
# Sau CLAHE: nếu còn cấu trúc → cảnh thật (kể cả thiếu sáng)
OBSTRUCTED_RECOVER_LAP_MIN = 28.0
OBSTRUCTED_RECOVER_EDGE_MIN = 0.015
# ====================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]
SKIN_REF = [10, 50, 280]

# ----- Error logging -----
def setup_error_logger():
    logger = logging.getLogger("drowsiness")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = setup_error_logger()


def beep_async(freq, dur):
    """winsound.Beep chan luong -> chay thread rieng."""
    def worker():
        try:
            winsound.Beep(freq, dur)
        except Exception as e:
            log.warning("Beep failed: %s", e)
    threading.Thread(target=worker, daemon=True).start()


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    v1 = euclidean_distance(pts[1], pts[5])
    v2 = euclidean_distance(pts[2], pts[4])
    hori = euclidean_distance(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hori + 1e-6)


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    return euclidean_distance(pts[2], pts[3]) / (euclidean_distance(pts[0], pts[1]) + 1e-6)


def compute_head_pitch_score(landmarks, w, h):
    nose, chin = landmarks[1], landmarks[152]
    le, re = landmarks[33], landmarks[263]
    eye_mid_y = ((le.y + re.y) / 2.0) * h
    denom = chin.y * h - eye_mid_y
    if abs(denom) < 1e-3:
        return 0.0
    return (nose.y * h - eye_mid_y) / denom


def compute_head_yaw_score(landmarks, w, h):
    nose, le, re = landmarks[1], landmarks[33], landmarks[263]
    eye_mid_x = ((le.x + re.x) / 2.0) * w
    eye_dist = abs((re.x - le.x) * w) + 1e-6
    return (nose.x * w - eye_mid_x) / (eye_dist / 2.0)


def center_roi_stats(gray):
    h, w = gray.shape[:2]
    y1, y2 = int(h * 0.18), int(h * 0.82)
    x1, x2 = int(w * 0.22), int(w * 0.78)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0
    return float(np.mean(roi >= 250)), float(np.mean(roi))


def enhance_lighting(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    clip_ratio, _ = center_roi_stats(gray)

    if mean_val < LOW_LIGHT_MEAN:
        gamma = 1.35
        table = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        ).astype("uint8")
        bright = cv2.LUT(bgr, table)
        lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        return out, "low_light"

    if clip_ratio > GLARE_CLIP_RATIO:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lf = l.astype(np.float32)
        hi = lf > 180
        lf[hi] = 180.0 + (lf[hi] - 180.0) * 0.40
        if clip_ratio > 0.20:
            lf *= 0.92
        l2 = np.clip(lf, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        return out, "glare"

    return bgr, "normal"


def roi_gray_mean(bgr, landmarks, indices, w, h, pad=3):
    xs = [int(landmarks[i].x * w) for i in indices]
    ys = [int(landmarks[i].y * h) for i in indices]
    if not xs:
        return None
    x1, x2 = max(0, min(xs) - pad), min(w - 1, max(xs) + pad)
    y1, y2 = max(0, min(ys) - pad), min(h - 1, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    g = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(np.mean(g))


def is_sunglasses_relative(bgr, landmarks, w, h, ratio_max):
    left_m = roi_gray_mean(bgr, landmarks, LEFT_EYE, w, h)
    right_m = roi_gray_mean(bgr, landmarks, RIGHT_EYE, w, h)
    skin_m = roi_gray_mean(bgr, landmarks, SKIN_REF, w, h, pad=6)
    if left_m is None or right_m is None or skin_m is None or skin_m < 1.0:
        return False
    eye_m = (left_m + right_m) / 2.0
    return (eye_m / skin_m) < ratio_max


def eyes_overexposed(bgr, landmarks, w, h):
    m1 = roi_gray_mean(bgr, landmarks, LEFT_EYE, w, h, pad=4)
    m2 = roi_gray_mean(bgr, landmarks, RIGHT_EYE, w, h, pad=4)
    if m1 is None or m2 is None:
        return False
    return ((m1 + m2) * 0.5) >= GLARE_EYE_MEAN_MIN

def is_color_obstructed(bgr):
    """
    Phát hiện che bằng vật liệu để ánh sáng xuyên qua (ngón tay, vải mỏng...)
    tạo ra màu đơn sắc áp đảo toàn khung hình — điển hình là cam/đỏ.
    Không cần quan tâm texture vì bản chất là một mảng màu gần như đồng nhất.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    h, w = h_ch.shape[:2]

    # Vùng trung tâm, tránh viền/góc bị vignette
    cy1, cy2 = int(h * 0.15), int(h * 0.85)
    cx1, cx2 = int(w * 0.15), int(w * 0.85)
    h_roi = h_ch[cy1:cy2, cx1:cx2].astype(np.float32)
    s_roi = s_ch[cy1:cy2, cx1:cx2].astype(np.float32)
    v_roi = v_ch[cy1:cy2, cx1:cx2].astype(np.float32)

    # OpenCV hue: 0-180. Cam-đỏ nằm khoảng 0-20 hoặc 160-180
    warm_mask = (h_roi <= 20) | (h_roi >= 160)
    warm_ratio = float(np.mean(warm_mask))

    mean_sat = float(np.mean(s_roi))
    mean_val = float(np.mean(v_roi))

    # Độ đồng nhất của hue trên TOÀN vùng trung tâm (không chỉ trong warm_mask)
    # — vật che ánh sáng xuyên qua cho hue rất đồng đều, cảnh thật (da người,
    # đồ vật màu cam trong nền) hiếm khi phủ kín + đồng đều như vậy
    hue_std = float(np.std(h_roi))

    is_warm_dominant = warm_ratio > 0.6
    is_saturated = mean_sat > 40
    is_uniform = hue_std < 25
    is_not_too_dark_or_bright = 10 < mean_val < 250

    return is_warm_dominant and is_saturated and is_uniform and is_not_too_dark_or_bright


def is_frame_obstructed(bgr):
    """
    Camera bị che gần hết — mọi màu.
    Hai bước:
      A) Flatness sau blur mạnh (bắt tay/vải/giấy, không cần màu đen).
      B) Structure recovery bằng CLAHE — phòng tối có mặt/cảnh sẽ lộ cạnh
         → KHÔNG obstructed; che ống kính thì CLAHE vẫn gần như phẳng.
    """
    # Check màu trước — rẻ và đặc trưng cho case ngón tay/vật che xuyên sáng
    if is_color_obstructed(bgr):
        return True

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return False

    # ----- B) Structure recovery (ưu tiên: thiếu sáng ≠ che) -----
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enh_blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    recover_lap = float(cv2.Laplacian(enh_blur, cv2.CV_64F).var())
    recover_edge = float(np.mean(cv2.Canny(enhanced, 40, 120) > 0))
    if (
        recover_lap >= OBSTRUCTED_RECOVER_LAP_MIN
        or recover_edge >= OBSTRUCTED_RECOVER_EDGE_MIN
    ):
        return False

    # ----- A) Flatness trên raw (che mọi màu) -----
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    lap_var = float(cv2.Laplacian(blur, cv2.CV_64F).var())

    k = OBSTRUCTED_LOCAL_KERNEL
    if k % 2 == 0:
        k += 1
    f = blur.astype(np.float32)
    mean = cv2.blur(f, (k, k))
    mean_sq = cv2.blur(f * f, (k, k))
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    low_var_ratio = float(np.mean(local_std < OBSTRUCTED_LOCAL_STD_MAX))

    cy1, cy2 = int(h * 0.20), int(h * 0.80)
    cx1, cx2 = int(w * 0.20), int(w * 0.80)
    center_std_map = local_std[cy1:cy2, cx1:cx2]
    center_std = float(np.mean(center_std_map)) if center_std_map.size else 999.0
    center = blur[cy1:cy2, cx1:cx2]
    center_lap = (
        float(cv2.Laplacian(center, cv2.CV_64F).var()) if center.size else lap_var
    )

    # Phải phẳng trên raw VÀ không phục hồi cấu trúc sau CLAHE
    if (
        low_var_ratio >= OBSTRUCTED_LOW_VAR_RATIO
        and lap_var < OBSTRUCTED_LAP_VAR_MAX
        and center_std < OBSTRUCTED_CENTER_STD_MAX
    ):
        return True

    if (
        center_lap < OBSTRUCTED_CENTER_LAP_MAX
        and center_std < OBSTRUCTED_CENTER_STD_MAX
        and low_var_ratio >= 0.60
        and lap_var < OBSTRUCTED_LAP_VAR_MAX * 1.35
    ):
        return True

    return False


def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["start_time", "end_time", "duration_sec", "alert_types"]
            )


def log_alert(start_ts, end_ts, reasons):
    try:
        ensure_log_header()
        dur = max(0.0, end_ts - start_ts)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S"),
                f"{dur:.1f}",
                " + ".join(reasons) if reasons else "",
            ])
    except Exception as e:
        log.error("log_alert failed: %s", e)


class DrowsinessApp:
    NAV = [
        ("monitor", "◉  Giám sát"),
        ("map", "⌖  Bản đồ & lộ trình"),
        ("assistant", "✦  Trợ lý AI"),
        ("history", "≡  Lịch sử cảnh báo"),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Driver Guard — Giám sát tài xế")
        self.root.geometry("1280x820")
        self.root.minsize(1120, 740)
        self.root.configure(fg_color=T.C["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.running = False
        self.thread = None
        self.cap = None
        self.landmarker = None
        self.photo = None
        self.frame_img = None
        self.frame_seq = 0
        self._shown_seq = -1
        self._video_size = (640, 480)
        self.rag_service = None

        self.ear = self.mar = self.pitch = self.yaw = 0.0
        self.nod_count = 0
        self.closed_dur = self.yawn_dur = self.deep_dur = 0.0
        self.fps = 0
        self.status = "STOPPED"
        self.reason = ""
        self.eye_progress = self.yawn_progress = 0.0
        self.env_flag = ""
        self.eye_unreliable = False
        self.current_th_display = EAR_THRESHOLD_NORMAL

        # Watchdog / system health
        self.last_heartbeat = 0.0
        self.system_error = False
        self.system_error_msg = ""
        self.detect_restarts = 0
        self.camera_obstructed = False
        self._obstructed_since = None

        # Phien lam viec
        self.session_start = None
        self.session_alerts = 0
        self._thanks_until = 0.0
        self._banner_state = None

        # Advisor (met moi -> goi y -> Gemini dien dat -> 2 nut)
        self._last_position = None
        self.health_vision = HealthVision()
        self.advisor = AdvisorService(
            get_position=self._advisor_position,
            get_route_points=self._advisor_route_points,
            get_planned_stops=self._advisor_planned_stops,
            get_health_signs=self.health_vision.active_signs,
        )
        self.advice_popup = None
        import trip_session
        self.trip = trip_session.TripSupervisor(self._trip_fix, self._trip_replan)

        self.ui = T.UiBridge(root)      # thread nen -> giao dien
        self.pages = {}
        self.nav_buttons = {}
        self.current_page = None
        self.planner = None

        self.intro_frame = ctk.CTkFrame(root, fg_color=T.C["bg"], corner_radius=0)
        self.main_frame = ctk.CTkFrame(root, fg_color=T.C["bg"], corner_radius=0)
        self.build_intro()
        self.build_main()
        self.show_intro()
        ensure_log_header()
        log.info("App started")
        if not self.health_vision.enabled:
            log.info("HealthVision tat: %s", self.health_vision.reason_disabled)
        self.root.after(1000, self._poll_advice)
        self.root.after(1500, self._poll_trip)
        self.root.after(1000, self._tick_slow)

    # ================================================================ MAN HINH GIOI THIEU
    def build_intro(self):
        f = self.intro_frame
        wrap = ctk.CTkFrame(f, fg_color="transparent")
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(wrap, text="DG", width=64, height=64, corner_radius=18,
                     fg_color=T.C["accent"], text_color="#04201c",
                     font=T.font(24, "bold")).pack()
        ctk.CTkLabel(wrap, text="Driver Guard", font=T.font(38, "bold"),
                     text_color=T.C["text"]).pack(pady=(16, 2))
        ctk.CTkLabel(wrap, text="Hệ thống giám sát dấu hiệu buồn ngủ của tài xế",
                     font=T.font(15), text_color=T.C["muted"]).pack(pady=(0, 26))

        grid = ctk.CTkFrame(wrap, fg_color="transparent")
        grid.pack()
        feats = [
            ("Phát hiện đa tín hiệu", "Mắt nhắm, ngáp, gật gù, cúi sâu giữ lâu"),
            ("Thích ứng môi trường", "Thiếu sáng, chói, kính râm, rung camera"),
            ("Tự giám sát độ tin cậy", "Watchdog, phát hiện camera bị che, ghi log"),
            ("Nhắc dừng nghỉ thông minh", "Gợi ý điểm dừng gần nhất, Gemini diễn đạt"),
        ]
        for i, (title, desc) in enumerate(feats):
            card = T.Card(grid, width=330, height=92)
            card.grid(row=i // 2, column=i % 2, padx=8, pady=8)
            card.pack_propagate(False)
            ctk.CTkLabel(card, text=title, font=T.font(14, "bold"), text_color=T.C["text"],
                         anchor="w").pack(fill="x", padx=18, pady=(16, 2))
            ctk.CTkLabel(card, text=desc, font=T.font(12), text_color=T.C["muted"],
                         anchor="w").pack(fill="x", padx=18)

        T.primary_button(wrap, "Vào ứng dụng", self.show_main, width=280, height=52
                         ).pack(pady=(26, 10))
        ctk.CTkLabel(
            wrap, text="Hệ thống chỉ nhận biết dấu hiệu quan sát được từ khuôn mặt, "
            "không khẳng định trạng thái bên trong và không thay thế chẩn đoán y tế.",
            font=T.font(11), text_color=T.C["muted"], wraplength=640, justify="center").pack()
        ctk.CTkLabel(f, text="Sáng tạo trẻ Quốc gia 2026  ·  Bảng C – AI",
                     font=T.font(11), text_color=T.C["idle"]).pack(side="bottom", pady=16)

    def show_intro(self):
        self.main_frame.pack_forget()
        self.intro_frame.pack(fill="both", expand=True)

    def show_main(self):
        self.intro_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)
        self.show_page("monitor")
        if not getattr(self, "_resume_checked", False):
            self._resume_checked = True
            self.root.after(1500, self._resume_trip_prompt)

    def back_to_intro(self):
        self.stop()
        self.show_intro()

    # ================================================================ KHUNG CHINH
    def build_main(self):
        m = self.main_frame
        m.grid_columnconfigure(1, weight=1)
        m.grid_rowconfigure(0, weight=1)

        # ---------- sidebar ----------
        side = ctk.CTkFrame(m, fg_color=T.C["surface"], corner_radius=0, width=244)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        side.pack_propagate(False)

        brand = ctk.CTkFrame(side, fg_color="transparent")
        brand.pack(fill="x", padx=18, pady=(22, 18))
        ctk.CTkLabel(brand, text="DG", width=42, height=42, corner_radius=12,
                     fg_color=T.C["accent"], text_color="#04201c",
                     font=T.font(16, "bold")).pack(side="left")
        names = ctk.CTkFrame(brand, fg_color="transparent")
        names.pack(side="left", padx=12)
        ctk.CTkLabel(names, text="Driver Guard", font=T.font(16, "bold"),
                     text_color=T.C["text"], anchor="w").pack(anchor="w")
        ctk.CTkLabel(names, text="Giám sát tài xế", font=T.font(11),
                     text_color=T.C["muted"], anchor="w").pack(anchor="w")

        for key, text in self.NAV:
            btn = ctk.CTkButton(
                side, text=text, anchor="w", height=44, corner_radius=10,
                font=T.font(13, "bold"), fg_color="transparent",
                hover_color=T.C["surface2"], text_color=T.C["muted"],
                command=lambda k=key: self.show_page(k))
            btn.pack(fill="x", padx=14, pady=2)
            self.nav_buttons[key] = btn

        # phan duoi sidebar (pack side=bottom: phan tu cuoi nam tren cung)
        T.ghost_button(side, "← Màn hình giới thiệu", self.back_to_intro
                       ).pack(side="bottom", fill="x", padx=16, pady=(6, 16))
        T.ghost_button(side, "Thoát ứng dụng", self.on_exit
                       ).pack(side="bottom", fill="x", padx=16, pady=(0, 0))

        sysbox = ctk.CTkFrame(side, fg_color=T.C["surface2"], corner_radius=12)
        sysbox.pack(side="bottom", fill="x", padx=14, pady=(0, 14))
        ctk.CTkLabel(sysbox, text="HỆ THỐNG", font=T.font(10, "bold"),
                     text_color=T.C["muted"], anchor="w").pack(fill="x", padx=14, pady=(12, 4))
        self.sys_rows = {}
        for key, label in (("camera", "Camera"), ("gemini", "Gemini AI"),
                           ("yolo", "YOLO"), ("poi", "Địa điểm"), ("trip", "Hành trình")):
            row = T.StatusRow(sysbox, label)
            row.pack(fill="x", padx=14, pady=3)
            self.sys_rows[key] = row
        ctk.CTkFrame(sysbox, height=8, fg_color="transparent").pack()

        # ---------- noi dung ----------
        self.content = ctk.CTkFrame(m, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=22, pady=18)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        for key, _ in self.NAV:
            page = ctk.CTkFrame(self.content, fg_color="transparent")
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page
        self.build_monitor(self.pages["monitor"])
        self.build_history(self.pages["history"])
        self.build_assistant(self.pages["assistant"])
        self._page_header(self.pages["map"], "Bản đồ & lộ trình",
                          "Lập tuyến, xem điểm dừng chân, trạm xăng, quán ăn và cơ sở y tế dọc đường")

    def _page_header(self, page, title, subtitle, right_builder=None):
        """Tao hang tieu de + tra ve (frame_tieu_de). Noi dung bat dau o row=1."""
        page.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        head.grid_columnconfigure(0, weight=1)
        titles = ctk.CTkFrame(head, fg_color="transparent")
        titles.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(titles, text=title, font=T.font(24, "bold"),
                     text_color=T.C["text"], anchor="w").pack(anchor="w")
        ctk.CTkLabel(titles, text=subtitle, font=T.font(12),
                     text_color=T.C["muted"], anchor="w").pack(anchor="w")
        if right_builder:
            right = ctk.CTkFrame(head, fg_color="transparent")
            right.grid(row=0, column=1, sticky="e")
            right_builder(right)
        return head

    def show_page(self, key):
        if key == "map" and self.planner is None:
            self._ensure_planner()
        self.pages[key].tkraise()
        self.current_page = key
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=T.C["surface2"], text_color=T.C["accent"])
            else:
                btn.configure(fg_color="transparent", text_color=T.C["muted"])
        if key == "history":
            self.refresh_history()
        elif key == "assistant":
            self._rag_boot()

    def _ensure_planner(self):
        if self.planner is not None:
            return self.planner
        page = self.pages["map"]
        holder = ctk.CTkFrame(page, fg_color="transparent")
        holder.grid(row=1, column=0, sticky="nsew")
        page.grid_rowconfigure(1, weight=1)
        self.planner = RoutePlanner(self.root, container=holder)
        self.planner.position_getter = self._agent_position
        self.planner.rag_getter = self._agent_rag
        from mini_monitor import MiniMonitor
        self.mini = MiniMonitor(self.planner.map_card, on_toggle=self.toggle)
        self.mini.attach()
        self.mini.set_running(self.running)
        self.mini.set_pill(self.status_pill.cget("text"), self.status_pill._kind)
        self.planner.trip_hooks = {"start": self.start_trip, "stop": self.stop_trip,
                                   "status": self.trip_status_text}
        return self.planner

    # ================================================================ THEO DOI HANH TRINH
    def _trip_fix(self):
        """Thread nen: vi tri + do chinh xac uoc luong theo nguon."""
        pos = route_map.auto_locate()
        if not pos:
            return None
        self._last_position = {"lat": pos["lat"], "lon": pos["lon"]}
        src = str(pos.get("source", ""))
        acc = 10000 if "IP" in src else 15 if "GPS" in src else 150
        return {"lat": pos["lat"], "lon": pos["lon"], "accuracy_m": acc, "source": src}

    def _trip_replan(self, session, fix, status):
        """Thread nen: tinh lai tuyen + ke hoach tu vi tri hien tai toi diem den."""
        import trip_planner as tp
        cur = {"lat": fix["lat"], "lon": fix["lon"], "name": "Vị trí hiện tại"}
        route = route_map.calculate_route(cur, session.destination)
        plan = tp.build_plan(
            route, lambda la, lo: route_map.find_places_near(la, lo, "all", tp.SEARCH_RADIUS_M),
            depart=datetime.datetime.now(), already_driven_min=session.cont_drive_min)
        session.origin = cur
        planner = self.planner
        if planner is not None:
            self.ui.post(lambda: planner.show_result({
                "origin": cur, "destination": session.destination, "route": route, "places": [],
                "places_warning": None, "plan": plan, "plan_text": tp.template_summary(plan),
                "plan_src": "template", "plan_error": ""}))
        return route, plan

    def start_trip(self, route, plan, origin, destination):
        import trip_session
        sess = trip_session.TripSession(route, plan, origin, destination)
        self.trip.start(sess)
        trip_session.save_state(sess)
        self.sys_rows["trip"].set("Đang theo dõi", "ok")

    def stop_trip(self):
        self.trip.stop(clear=True)
        self.sys_rows["trip"].set("Chưa bắt đầu", "idle")

    def trip_status_text(self):
        st = self.trip.status
        if not self.trip.active:
            return "Chưa có hành trình đang theo dõi."
        if not st:
            return "Đang chờ vị trí đầu tiên…"
        import trip_planner as tp
        s = (f"Còn {st['remaining_km']:.0f} km, dự kiến đến {st['eta'].strftime('%H:%M')}. "
             f"Lái liên tục {tp.fmt_dur(st['cont_drive_min'])}.")
        n = st["next_stop"]
        if n:
            s += f" Điểm dừng kế tiếp: {n['name']}, cách {tp.fmt_km(n['dist_m'])}."
        if st["mode"] == "time":
            s += " (Ước tính theo thời gian — cần GPS thật để chính xác.)"
        return s

    def _poll_trip(self):
        """Main thread: su kien hanh trinh -> popup (muc >=1) hoac cap nhat dong trang thai."""
        try:
            while True:
                ev = self.trip.events.get_nowait()
                if ev["level"] >= 1 and self.advice_popup is None:
                    n = (self.trip.status or {}).get("next_stop")
                    target = None
                    if ev.get("stop"):
                        s = ev["stop"]
                        target = {"lat": s["lat"], "lon": s["lon"], "name": s["name"]}
                    elif n and n.get("known"):
                        target = {"lat": n["lat"], "lon": n["lon"], "name": n["name"]}
                    self.advice_popup = AdvicePopup(
                        self.root, {"level": max(1, ev["level"]), "level_text": "", "text": ev["text"],
                                    "source": "trip", "target": target, "position": ev.get("position"),
                                    "signs": []},
                        on_navigate=self._navigate_to_advice, on_close=self._on_advice_closed)
                elif ev["level"] == 0 and ev.get("text"):
                    self._thanks_until = 0
                    self._set_banner(ev["text"], "ok" if ev["type"] != "arrived" else "ok")
                if ev["type"] == "arrived":
                    self.sys_rows["trip"].set("Đã đến nơi", "ok")
        except queue.Empty:
            pass
        except Exception as e:
            log.warning("_poll_trip: %s", e)
        self.root.after(1000, self._poll_trip)

    def _resume_trip_prompt(self):
        import trip_session
        sess = trip_session.load_state()
        if sess is None or self.trip.active:
            return
        if messagebox.askyesno("Hành trình dang dở",
                               "Phát hiện một hành trình chưa hoàn thành. Tiếp tục theo dõi?",
                               parent=self.root):
            self.trip.start(sess)
            self.sys_rows["trip"].set("Đang theo dõi", "ok")
            planner = self._ensure_planner()
            import trip_planner as tp
            planner.show_result({"origin": sess.origin or {"lat": sess.route["points"][0][0], "lon": sess.route["points"][0][1]},
                                 "destination": sess.destination or {"lat": sess.route["points"][-1][0], "lon": sess.route["points"][-1][1]},
                                 "route": sess.route, "places": [], "places_warning": None, "plan": sess.plan,
                                 "plan_text": tp.template_summary(sess.plan), "plan_src": "template", "plan_error": ""})
        else:
            trip_session.clear_state()

    def _agent_position(self):
        """Vi tri cho Agent (chay trong thread nen): dung vi tri gan nhat, neu chua co thi dinh vi."""
        return self._last_position or self._advisor_position()

    def _agent_rag(self):
        if self.rag_service is None:
            try:
                self.rag_service = RAGService()
            except Exception:
                return None
        return self.rag_service

    # ================================================================ TRANG GIAM SAT
    def build_monitor(self, page):
        page.grid_rowconfigure(2, weight=1)

        def right(h):
            self.status_pill = T.Pill(h, "ĐÃ DỪNG", "idle")
            self.status_pill.pack(side="left", padx=(0, 14))
            self.btn_toggle = ctk.CTkButton(
                h, text="▶  Bắt đầu giám sát", width=210, height=44, corner_radius=12,
                font=T.font(14, "bold"), fg_color=T.C["ok"], hover_color="#16a34a",
                text_color="#04150b", command=self.toggle)
            self.btn_toggle.pack(side="left")

        self._page_header(page, "Giám sát tài xế",
                          "Phân tích khuôn mặt theo thời gian thực — hoạt động ngoại tuyến", right)

        # banner trang thai
        self.banner = ctk.CTkFrame(page, fg_color=T.C["surface"], corner_radius=12,
                                   border_width=1, border_color=T.C["border"], height=52)
        self.banner.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        self.banner.grid_propagate(False)
        self.banner_label = ctk.CTkLabel(
            self.banner, text="", font=T.font(14, "bold"), text_color=T.C["muted"], anchor="w")
        self.banner_label.pack(fill="both", expand=True, padx=18)
        self._set_banner("Nhấn «Bắt đầu giám sát» để khởi động camera.", "idle")

        body = ctk.CTkFrame(page, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=1)

        # ---- khung video ----
        vcard = T.Card(body)
        vcard.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.video_holder = ctk.CTkFrame(vcard, fg_color="#05080b", corner_radius=10)
        self.video_holder.pack(fill="both", expand=True, padx=10, pady=(10, 8))
        self.video_label = tk.Label(
            self.video_holder, bg="#05080b", fg=T.C["muted"], bd=0,
            font=("Segoe UI", 13), text="Camera đang tắt\nNhấn «Bắt đầu giám sát»")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_holder.bind("<Configure>", self._on_video_resize)

        strip = ctk.CTkFrame(vcard, fg_color="transparent")
        strip.pack(fill="x", padx=10, pady=(0, 10))
        strip.grid_columnconfigure((0, 1, 2), weight=1, uniform="s")
        self.tile_time = T.Tile(strip, "THỜI GIAN GIÁM SÁT", "00:00:00")
        self.tile_alerts = T.Tile(strip, "CẢNH BÁO TRONG PHIÊN", "0")
        self.tile_fps = T.Tile(strip, "TỐC ĐỘ XỬ LÝ", "— FPS")
        for i, t in enumerate((self.tile_time, self.tile_alerts, self.tile_fps)):
            t.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 5, 0 if i == 2 else 5))

        # ---- cot phai ----
        col = ctk.CTkFrame(body, fg_color="transparent", width=360)
        col.grid(row=0, column=1, sticky="ns")
        col.grid_propagate(False)

        fcard = T.Card(col, title="Mức độ mệt mỏi (10 phút gần nhất)")
        fcard.pack(fill="x", pady=(0, 14))
        self.fatigue_level = ctk.CTkLabel(fcard, text="Ổn định", font=T.font(26, "bold"),
                                          text_color=T.C["ok"], anchor="w")
        self.fatigue_level.pack(fill="x", padx=16)
        self.fatigue_bar = ctk.CTkProgressBar(fcard, height=10, corner_radius=5,
                                              progress_color=T.C["ok"], fg_color=T.C["surface2"])
        self.fatigue_bar.set(0)
        self.fatigue_bar.pack(fill="x", padx=16, pady=(8, 8))
        self.fatigue_signs = ctk.CTkLabel(
            fcard, text="Chưa ghi nhận dấu hiệu bất thường.", font=T.font(12),
            text_color=T.C["muted"], anchor="w", justify="left", wraplength=320)
        self.fatigue_signs.pack(fill="x", padx=16, pady=(0, 14))

        mcard = T.Card(col, title="Chỉ số thời gian thực")
        mcard.pack(fill="x")
        grid = ctk.CTkFrame(mcard, fg_color="transparent")
        grid.pack(fill="x", padx=12)
        grid.grid_columnconfigure((0, 1), weight=1, uniform="m")
        self.tiles = {}
        for i, (key, label) in enumerate([
            ("ear", "EAR (MẮT)"), ("mar", "MAR (MIỆNG)"), ("pitch", "PITCH"), ("yaw", "YAW"),
            ("nod", "GẬT GÙ"), ("deep", "CÚI SÂU"), ("closed", "NHẮM MẮT"), ("yawn", "NGÁP"),
        ]):
            t = T.Tile(grid, label, compact=True)
            t.grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=3)
            self.tiles[key] = t

        for text, attr in (("Tiến độ nhắm mắt", "eye_bar"), ("Tiến độ ngáp", "yawn_bar")):
            ctk.CTkLabel(mcard, text=text, font=T.font(11), text_color=T.C["muted"],
                         anchor="w").pack(fill="x", padx=16, pady=(8, 2))
            bar = ctk.CTkProgressBar(mcard, height=8, corner_radius=4,
                                     progress_color=T.C["info"], fg_color=T.C["surface2"])
            bar.set(0)
            bar.pack(fill="x", padx=16)
            setattr(self, attr, bar)
        ctk.CTkFrame(mcard, height=12, fg_color="transparent").pack()

    def _on_video_resize(self, event):
        w, h = max(event.width - 8, 160), max(event.height - 8, 120)
        if w * 3 > h * 4:
            w = int(h * 4 / 3)
        else:
            h = int(w * 3 / 4)
        self._video_size = (w, h)

    def _set_banner(self, text, kind):
        if self._banner_state == (text, kind):
            return
        self._banner_state = (text, kind)
        fg, border, color = {
            "idle": (T.C["surface"], T.C["border"], T.C["muted"]),
            "ok": (T.C["ok_bg"], T.C["ok"], T.C["ok"]),
            "warn": (T.C["warn_bg"], T.C["warn"], T.C["warn"]),
            "danger": (T.C["danger_bg"], T.C["danger"], "#ff8a8a"),
        }[kind]
        self.banner.configure(fg_color=fg, border_color=border)
        self.banner_label.configure(text=text, text_color=color)

    def _pill(self, text, kind, detail=""):
        self.status_pill.set(text, kind)
        if getattr(self, "mini", None) is not None:
            self.mini.set_pill(text, kind, detail)

    def _set_running_ui(self, running):
        if getattr(self, "mini", None) is not None:
            self.mini.set_running(running)
        if running:
            self.btn_toggle.configure(text="■  Dừng giám sát", fg_color=T.C["danger"],
                                      hover_color="#dc2626", text_color="white")
        else:
            self.btn_toggle.configure(text="▶  Bắt đầu giám sát", fg_color=T.C["ok"],
                                      hover_color="#16a34a", text_color="#04150b")

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    # ================================================================ TRANG LICH SU
    def build_history(self, page):
        page.grid_rowconfigure(2, weight=1)
        self._page_header(page, "Lịch sử cảnh báo", "Nhật ký các lần hệ thống phát cảnh báo",
                          lambda h: T.ghost_button(h, "Làm mới", self.refresh_history, width=110
                                                   ).pack())
        stats = ctk.CTkFrame(page, fg_color="transparent")
        stats.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        stats.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="h")
        self.hist_tiles = {}
        for i, (key, label) in enumerate([("total", "TỔNG SỐ CẢNH BÁO"), ("today", "HÔM NAY"),
                                          ("dur", "TỔNG THỜI LƯỢNG"), ("top", "LOẠI THƯỜNG GẶP")]):
            t = T.Tile(stats, label)
            t.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0 if i == 3 else 6))
            self.hist_tiles[key] = t

        card = T.Card(page)
        card.grid(row=2, column=0, sticky="nsew")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("DG.Treeview", background=T.C["surface"], fieldbackground=T.C["surface"],
                        foreground=T.C["text"], rowheight=34, borderwidth=0, font=("Segoe UI", 11))
        style.configure("DG.Treeview.Heading", background=T.C["surface2"], foreground=T.C["muted"],
                        borderwidth=0, font=("Segoe UI", 10, "bold"), padding=8)
        style.map("DG.Treeview", background=[("selected", T.C["info"])],
                  foreground=[("selected", "white")])
        style.map("DG.Treeview.Heading", background=[("active", T.C["surface2"])])
        style.layout("DG.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=8, pady=8)
        self.hist_tree = ttk.Treeview(
            inner, columns=("time", "dur", "types"), show="headings", style="DG.Treeview")
        for col, text, width, anchor in (("time", "Thời điểm", 200, "w"),
                                         ("dur", "Kéo dài (giây)", 130, "center"),
                                         ("types", "Loại cảnh báo", 400, "w")):
            self.hist_tree.heading(col, text=text, anchor="w" if anchor == "w" else "center")
            self.hist_tree.column(col, width=width, anchor=anchor)
        sb = ctk.CTkScrollbar(inner, command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.hist_tree.pack(side="left", fill="both", expand=True)
        self.hist_empty = ctk.CTkLabel(card, text="", font=T.font(13), text_color=T.C["muted"])

    def refresh_history(self):
        rows = []
        try:
            with open(LOG_PATH, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for r in reader:
                    if len(r) >= 4:
                        rows.append(r)
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning("refresh_history: %s", e)

        self.hist_tree.delete(*self.hist_tree.get_children())
        today = datetime.now().strftime("%Y-%m-%d")
        total_dur, today_n, counts = 0.0, 0, {}
        for r in rows:
            try:
                total_dur += float(r[2])
            except ValueError:
                pass
            if r[0].startswith(today):
                today_n += 1
            for kind in [k.strip() for k in r[3].split("+") if k.strip()]:
                counts[kind] = counts.get(kind, 0) + 1
        for r in reversed(rows[-300:]):
            self.hist_tree.insert("", "end", values=(r[0], r[2], r[3] or "—"))

        self.hist_tiles["total"].set(str(len(rows)))
        self.hist_tiles["today"].set(str(today_n))
        self.hist_tiles["dur"].set(f"{int(total_dur // 60)} phút {int(total_dur % 60)} giây")
        self.hist_tiles["top"].set(max(counts, key=counts.get) if counts else "—")
        if rows:
            self.hist_empty.place_forget()
        else:
            self.hist_empty.configure(text="Chưa có cảnh báo nào được ghi nhận.")
            self.hist_empty.place(relx=0.5, rely=0.5, anchor="center")

    # ================================================================ TRANG TRO LY AI (RAG v2)
    SUGGESTIONS = [
        "Khi đang lái xe thấy buồn ngủ thì nên làm gì?",
        "Những dấu hiệu nào cho thấy tôi đang buồn ngủ?",
        "Hệ thống cảnh báo khi nào?",
        "Lái xe liên tục tối đa bao lâu thì phải nghỉ?",
    ]

    def build_assistant(self, page):
        self.rag_history = []
        self.rag_busy = False
        self._rag_booted = False
        self._rag_sources = []
        page.grid_rowconfigure(1, weight=1)

        def right(h):
            self.rag_pill = T.Pill(h, "Đang khởi động…", "idle")
            self.rag_pill.pack(side="left", padx=(0, 12))
            T.ghost_button(h, "Cuộc trò chuyện mới", self._rag_new_chat, width=180).pack(side="left")

        self._page_header(page, "Trợ lý an toàn AI",
                          "Hỏi đáp dựa trên tài liệu của bạn, có trích dẫn nguồn", right)

        body = ctk.CTkFrame(page, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ---- khung chat ----
        chat = T.Card(body)
        chat.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        chat.grid_columnconfigure(0, weight=1)
        chat.grid_rowconfigure(1, weight=1)

        self.rag_chips = ctk.CTkFrame(chat, fg_color="transparent")
        self.rag_chips.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 0))
        ctk.CTkLabel(self.rag_chips, text="Gợi ý câu hỏi", font=T.font(11, "bold"),
                     text_color=T.C["muted"]).pack(anchor="w", pady=(0, 6))
        for q in self.SUGGESTIONS:
            ctk.CTkButton(
                self.rag_chips, text=q, anchor="w", height=34, corner_radius=17,
                font=T.font(12), fg_color=T.C["surface2"], hover_color=T.C["border"],
                text_color=T.C["text"], command=lambda t=q: self._rag_ask(t)
            ).pack(fill="x", pady=2)

        self.rag_chat = ctk.CTkTextbox(
            chat, wrap="word", fg_color=T.C["surface"], text_color=T.C["text"],
            font=T.font(13), corner_radius=10, activate_scrollbars=True)
        self.rag_chat.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        tb = self.rag_chat._textbox
        base = tkfont.Font(font=tb.cget("font"))
        bold = base.copy()
        bold.configure(weight="bold")
        self._rag_fonts = (base, bold)           # giu tham chieu de khong bi huy
        tb.tag_configure("who_user", foreground=T.C["accent"], font=bold, spacing1=10)
        tb.tag_configure("who_bot", foreground=T.C["info"], font=bold, spacing1=10)
        tb.tag_configure("b", font=bold)
        tb.tag_configure("cite", foreground="#60a5fa", font=bold)
        tb.tag_configure("sys", foreground=T.C["muted"])
        tb.configure(spacing3=3)
        self.rag_chat.configure(state="disabled")

        self.rag_status_var = ctk.StringVar(value="")
        ctk.CTkLabel(chat, textvariable=self.rag_status_var, font=T.font(11),
                     text_color=T.C["muted"], anchor="w").grid(row=2, column=0, sticky="ew", padx=16)

        bottom = ctk.CTkFrame(chat, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 12))
        bottom.grid_columnconfigure(0, weight=1)
        self.rag_question_var = ctk.StringVar()
        self.rag_entry = ctk.CTkEntry(
            bottom, textvariable=self.rag_question_var, height=46, corner_radius=12,
            placeholder_text="Nhập câu hỏi về an toàn khi lái xe…", font=T.font(13),
            fg_color=T.C["surface2"], border_color=T.C["border"], text_color=T.C["text"])
        self.rag_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.rag_entry.bind("<Return>", lambda _e: self._rag_ask())
        self.rag_ask_btn = T.primary_button(bottom, "Gửi", self._rag_ask, width=100, height=46)
        self.rag_ask_btn.grid(row=0, column=1)

        # ---- cot phai: nguon + kho tai lieu ----
        side = ctk.CTkFrame(body, fg_color="transparent", width=340)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.grid_rowconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=1)
        side.grid_columnconfigure(0, weight=1)

        src_card = T.Card(side, title="Nguồn tham chiếu")
        src_card.grid(row=0, column=0, sticky="nsew", pady=(0, 14))
        self.rag_src_frame = ctk.CTkScrollableFrame(src_card, fg_color="transparent", height=110)
        self.rag_src_frame.pack(fill="x", padx=6)
        self.rag_src_hint = ctk.CTkLabel(
            self.rag_src_frame, text="Nguồn của câu trả lời sẽ hiện ở đây.",
            font=T.font(12), text_color=T.C["muted"], anchor="w", wraplength=270, justify="left")
        self.rag_src_hint.pack(fill="x", padx=8, pady=6)
        self.rag_preview = ctk.CTkTextbox(
            src_card, height=120, wrap="word", fg_color=T.C["surface2"], text_color=T.C["muted"],
            font=T.font(11), corner_radius=8)
        self.rag_preview.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        self.rag_preview.configure(state="disabled")

        doc_card = T.Card(side, title="Kho tài liệu")
        doc_card.grid(row=1, column=0, sticky="nsew")
        lb_wrap = ctk.CTkFrame(doc_card, fg_color="transparent")
        lb_wrap.pack(fill="both", expand=True, padx=12)
        self.rag_doc_list = tk.Listbox(
            lb_wrap, font=("Segoe UI", 10), bg=T.C["surface2"], fg=T.C["text"],
            selectbackground=T.C["info"], selectforeground="white", highlightthickness=0,
            relief="flat", activestyle="none", borderwidth=0, height=5)
        self.rag_doc_list.pack(fill="both", expand=True)
        self.rag_doc_info = ctk.CTkLabel(doc_card, text="", font=T.font(11),
                                         text_color=T.C["muted"], anchor="w")
        self.rag_doc_info.pack(fill="x", padx=14, pady=(6, 0))
        btns = ctk.CTkFrame(doc_card, fg_color="transparent")
        btns.pack(fill="x", padx=12, pady=(8, 12))
        btns.grid_columnconfigure((0, 1), weight=1, uniform="d")
        self.rag_add_btn = T.ghost_button(btns, "Thêm tài liệu…", self._rag_add_files)
        self.rag_add_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.rag_ingest_btn = T.ghost_button(btns, "Nạp / cập nhật", self._rag_ingest)
        self.rag_ingest_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.rag_del_btn = T.ghost_button(btns, "Xóa tài liệu đã chọn", self._rag_delete_selected)
        self.rag_del_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self._rag_write("Hệ thống\n", "who_bot")
        self._rag_write("Mình trả lời dựa trên tài liệu trong kho và ghi rõ nguồn. "
                        "Đây là công cụ tham khảo, không thay thế cảnh báo thời gian thực "
                        "và không phải tư vấn y tế hay pháp lý.\n", "sys")

    # ---------- tien ich chat ----------
    def _rag_write(self, text, *tags):
        try:
            self.rag_chat.configure(state="normal")
            self.rag_chat._textbox.insert("end", text, tags)
            self.rag_chat.configure(state="disabled")
            self.rag_chat._textbox.see("end")
        except Exception:
            pass

    def _rag_write_markdown(self, text):
        for line in text.split("\n"):
            line = re.sub(r"^\s*[*-]\s+", "• ", line)
            pos = 0
            for m in re.finditer(r"\*\*(.+?)\*\*|\[(\d{1,2})\]", line):
                self._rag_write(line[pos:m.start()])
                self._rag_write(m.group(1) if m.group(1) else m.group(0),
                                "b" if m.group(1) else "cite")
                pos = m.end()
            self._rag_write(line[pos:] + "\n")

    def _rag_set_busy(self, busy, status=""):
        self.rag_busy = busy
        state = "disabled" if busy else "normal"
        for w in (self.rag_ask_btn, self.rag_add_btn, self.rag_ingest_btn, self.rag_del_btn):
            w.configure(state=state)
        self.rag_status_var.set(status)

    def _rag_progress(self, message):
        self.ui.post(lambda m=message: self.rag_status_var.set(m))

    def _rag_new_chat(self):
        if self.rag_busy:
            return
        self.rag_history.clear()
        self.rag_chat.configure(state="normal")
        self.rag_chat._textbox.delete("1.0", "end")
        self.rag_chat.configure(state="disabled")
        self._rag_show_sources([])
        self.rag_chips.grid()
        self._rag_write("Hệ thống\n", "who_bot")
        self._rag_write("Đã bắt đầu cuộc trò chuyện mới.\n", "sys")

    # ---------- khoi dong / kho tai lieu ----------
    def _get_rag_service(self):
        if RAGService is None:
            raise RuntimeError("Không import được rag_gemini.py (" + RAG_IMPORT_ERROR +
                               "). Hãy chạy: pip install -r requirements.txt")
        if self.rag_service is None:
            self.rag_service = RAGService()
        return self.rag_service

    def _rag_boot(self):
        """Lan dau mo trang: tao dich vu va dong bo tai lieu o thread rieng."""
        if self._rag_booted:
            return
        self._rag_booted = True
        self._rag_set_busy(True, "Đang khởi động trợ lý…")

        def worker():
            msg = ""
            try:
                svc = self._get_rag_service()
                self.ui.post(self._rag_update_status)
                if svc.online:
                    res = svc.ingest_folder(progress=self._rag_progress)
                    if res["added"] or res["updated"] or res["removed"]:
                        msg = (f"Đã đồng bộ: +{res['added']} mới, {res['updated']} cập nhật, "
                               f"{res['removed']} xóa.")
                    if res["errors"]:
                        msg = "Lưu ý: " + "; ".join(res["errors"][:2])
            except Exception as exc:
                msg = f"Lỗi khởi động: {type(exc).__name__}: {exc}"
                log.warning("RAG boot: %s", exc)
            self.ui.post(lambda m=msg: self._rag_boot_done(m))

        threading.Thread(target=worker, daemon=True).start()

    def _rag_boot_done(self, message):
        self._rag_set_busy(False, message)
        self._rag_update_status()
        self._rag_refresh_docs()

    def _rag_update_status(self):
        svc = self.rag_service
        if svc is None:
            self.rag_pill.set("Chưa sẵn sàng", "danger")
        elif svc.online:
            self.rag_pill.set("Gemini sẵn sàng", "ok")
        else:
            self.rag_pill.set("Ngoại tuyến · chỉ tìm từ khóa", "warn")

    def _rag_refresh_docs(self):
        svc = self.rag_service
        self.rag_doc_list.delete(0, "end")
        if svc is None:
            return
        try:
            docs = svc.list_documents()
            for d in docs:
                self.rag_doc_list.insert("end", f"{d['source']}  ·  {d['chunks']} đoạn")
            self.rag_doc_info.configure(
                text=f"{len(docs)} tài liệu · {svc.collection.count()} đoạn"
                if docs else "Chưa có tài liệu. Bấm «Thêm tài liệu…»")
        except Exception as e:
            log.warning("_rag_refresh_docs: %s", e)

    def _rag_run_ingest(self, status):
        self._rag_set_busy(True, status)

        def worker():
            try:
                svc = self._get_rag_service()
                res = svc.ingest_folder(progress=self._rag_progress)
                msg = (f"Nạp xong: +{res['added']} mới, {res['updated']} cập nhật, "
                       f"{res['unchanged']} không đổi, {res['removed']} xóa.")
                if res["errors"]:
                    msg += "  Lỗi: " + "; ".join(res["errors"][:2])
            except Exception as exc:
                msg = f"Lỗi nạp tài liệu: {type(exc).__name__}: {exc}"
            self.ui.post(lambda m=msg: self._rag_ingest_done(m))

        threading.Thread(target=worker, daemon=True).start()

    def _rag_ingest(self):
        if self.rag_busy:
            return
        self._rag_run_ingest("Đang nạp tài liệu…")

    def _rag_ingest_done(self, message):
        self._rag_set_busy(False, message)
        self._rag_update_status()
        self._rag_refresh_docs()

    def _rag_add_files(self):
        if self.rag_busy:
            return
        paths = filedialog.askopenfilenames(
            parent=self.root, title="Chọn tài liệu",
            filetypes=[("Tài liệu", "*.pdf *.docx *.txt *.md"), ("Tất cả", "*.*")])
        if not paths:
            return
        try:
            added = self._get_rag_service().add_files(list(paths))
        except Exception as exc:
            messagebox.showerror("Không thêm được", str(exc), parent=self.root)
            return
        if not added:
            messagebox.showinfo("Không có tệp hợp lệ", "Chỉ hỗ trợ PDF, DOCX, TXT, MD.", parent=self.root)
            return
        self._rag_run_ingest(f"Đã thêm {len(added)} tệp, đang nạp…")

    def _rag_delete_selected(self):
        if self.rag_busy:
            return
        sel = self.rag_doc_list.curselection()
        if not sel:
            messagebox.showinfo("Chưa chọn", "Hãy chọn một tài liệu trong danh sách.", parent=self.root)
            return
        source = self.rag_doc_list.get(sel[0]).split("  ·  ")[0]
        if not messagebox.askyesno(
                "Xóa tài liệu", f"Xóa «{source}» khỏi kho và xóa tệp trong thư mục documents?",
                parent=self.root):
            return
        self._rag_set_busy(True, "Đang xóa…")

        def worker():
            try:
                self._get_rag_service().remove_document(source)
                msg = f"Đã xóa {source}."
            except Exception as exc:
                msg = f"Lỗi xóa: {type(exc).__name__}: {exc}"
            self.ui.post(lambda m=msg: self._rag_ingest_done(m))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- hoi dap ----------
    def _rag_ask(self, preset=None):
        if self.rag_busy:
            return
        question = (preset or self.rag_question_var.get()).strip()
        if not question:
            return
        self.rag_question_var.set("")
        self.rag_chips.grid_remove()
        self._rag_write("Bạn\n", "who_user")
        self._rag_write(question + "\n")
        self._rag_write("Trợ lý\n", "who_bot")
        tb = self.rag_chat._textbox
        tb.mark_set("rag_start", "end-1c")
        tb.mark_gravity("rag_start", "left")
        self._rag_set_busy(True, "Đang tìm trong kho tài liệu…")
        history = list(self.rag_history)
        streamed = {"n": 0}

        def on_token(piece):
            streamed["n"] += 1
            self.ui.post(lambda t=piece: self._rag_write(t))

        def worker():
            try:
                svc = self._get_rag_service()
                result = svc.ask(question, history=history, progress=self._rag_progress,
                                 on_token=on_token)
            except Exception as exc:
                result = {"answer": f"Không thể trả lời: {type(exc).__name__}: {exc}",
                          "sources": [], "mode": "error", "query": question}
            self.ui.post(lambda r=result, st=streamed["n"]: self._rag_answer_done(question, r))

        threading.Thread(target=worker, daemon=True).start()

    def _rag_answer_done(self, question, result):
        tb = self.rag_chat._textbox
        self.rag_chat.configure(state="normal")
        try:
            tb.delete("rag_start", "end-1c")       # bo ban nhap dang stream, thay bang ban da dinh dang
        except Exception:
            pass
        self.rag_chat.configure(state="disabled")
        self._rag_write_markdown(result["answer"])
        mode = result["mode"]
        n_src = len(result["sources"])
        meta = {
            "ai": f"Gemini · {n_src} nguồn",
            "offline": "Ngoại tuyến · hiển thị đoạn trích, chưa tổng hợp",
            "none": "Ngoài phạm vi tài liệu hiện có",
            "empty": "Kho tài liệu trống",
        }.get(mode, "")
        if mode == "ai" and result.get("query") and result["query"] != question:
            meta += f"  ·  hiểu là: «{result['query']}»"
        if meta:
            self._rag_write(meta + "\n", "sys")
        if mode == "ai":
            self.rag_history.append((question, result["answer"]))
            self.rag_history[:] = self.rag_history[-6:]
        self._rag_show_sources(result["sources"])
        self._rag_set_busy(False, "")
        self._rag_update_status()

    def _rag_show_sources(self, sources):
        for w in list(self.rag_src_frame.winfo_children()):
            w.destroy()
        self._rag_sources = sources
        self._rag_preview_set("")
        if not sources:
            ctk.CTkLabel(self.rag_src_frame, text="Nguồn của câu trả lời sẽ hiện ở đây.",
                         font=T.font(12), text_color=T.C["muted"], anchor="w",
                         wraplength=270, justify="left").pack(fill="x", padx=8, pady=6)
            return
        for s in sources:
            loc = s["source"] + (f" · tr.{s['page']}" if s["page"] else "")
            head = f"[{s['n']}] {loc}"
            if s.get("heading"):
                head += "\n" + s["heading"]
            ctk.CTkButton(
                self.rag_src_frame, text=head, anchor="w", height=44, corner_radius=8,
                font=T.font(11), fg_color=T.C["surface2"] if s["cited"] else "transparent",
                hover_color=T.C["border"],
                text_color=T.C["text"] if s["cited"] else T.C["muted"],
                command=lambda x=s: self._rag_preview_set(x["text"])
            ).pack(fill="x", padx=4, pady=2)
        first = next((s for s in sources if s["cited"]), sources[0])
        self._rag_preview_set(first["text"])

    def _rag_preview_set(self, text):
        self.rag_preview.configure(state="normal")
        self.rag_preview.delete("1.0", "end")
        self.rag_preview.insert("1.0", text)
        self.rag_preview.configure(state="disabled")

    # ================================================================ ADVISOR (vi tri / tuyen / popup)
    def _advisor_position(self):
        """Chay trong thread advisor (duoc phep cham): Windows -> GPS -> IP."""
        pos = route_map.auto_locate()
        if pos:
            self._last_position = {"lat": pos["lat"], "lon": pos["lon"]}
        return self._last_position

    def _advisor_planned_stops(self):
        planner = self.planner
        try:
            if planner is not None and planner.plan:
                import trip_planner
                return trip_planner.stops_as_places(planner.plan)
        except Exception:
            pass
        return []

    def _advisor_route_points(self):
        planner = self.planner
        try:
            if planner is not None and planner.route:
                return planner.route["points"]
        except Exception:
            pass
        return []

    def _poll_advice(self):
        """Main thread: nhan loi nhac tu advisor va hien popup."""
        try:
            while True:
                advice = self.advisor.out_queue.get_nowait()
                self.trip.notify_fatigue(advice.get("level", 0))    # theo doi hanh trinh chat hon
                if self.advice_popup is not None:
                    continue
                self.advice_popup = AdvicePopup(
                    self.root, advice,
                    on_navigate=self._navigate_to_advice,
                    on_close=self._on_advice_closed,
                )
        except queue.Empty:
            pass
        except Exception as e:
            log.warning("_poll_advice: %s", e)
        self.root.after(500, self._poll_advice)

    def _on_advice_closed(self):
        self.advice_popup = None
        self._thanks_until = time.time() + 6.0   # loi cam on co dinh, khong goi API

    def _navigate_to_advice(self, target, position):
        self.show_page("map")
        self.planner.route_to(target["lat"], target["lon"], target.get("name", ""),
                              origin=position)

    # ================================================================ 1 Hz: trang thai he thong
    def _tick_slow(self):
        try:
            # --- he thong ---
            if self.system_error:
                self.sys_rows["camera"].set("Lỗi", "danger")
            elif self.camera_obstructed and self.running:
                self.sys_rows["camera"].set("Bị che", "warn")
            elif self.running:
                self.sys_rows["camera"].set("Hoạt động", "ok")
            else:
                self.sys_rows["camera"].set("Tắt", "idle")

            if self.advisor.phraser.client is not None:
                self.sys_rows["gemini"].set("Sẵn sàng", "ok")
            else:
                self.sys_rows["gemini"].set("Ngoại tuyến", "warn")
            if self.trip.active:
                st = self.trip.status
                self.sys_rows["trip"].set(
                    f"Còn {st['remaining_km']:.0f} km" if st else "Chờ vị trí", "ok")
            if self.health_vision.enabled:
                self.sys_rows["yolo"].set("Đang bật", "ok")
            else:
                self.sys_rows["yolo"].set("Chưa bật", "idle")
            prov = poi_provider.configured_provider()
            st = poi_provider.last_status()
            if prov == "overpass":
                self.sys_rows["poi"].set("Overpass*", "warn")
            elif st.get("error") and st.get("provider") != prov:
                self.sys_rows["poi"].set(prov.capitalize() + " lỗi", "danger")
            else:
                self.sys_rows["poi"].set(prov.capitalize(), "ok")

            # --- phien lam viec ---
            if self.running and self.session_start:
                secs = int(time.time() - self.session_start)
                self.tile_time.set(f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}")
            self.tile_alerts.set(str(self.session_alerts))

            # --- muc do met moi ---
            if self.running:
                info = self.advisor.scorer.evaluate(self.health_vision.active_signs())
                lv = info["level"]
                text, color = {
                    0: ("Ổn định", T.C["ok"]), 1: ("Nhắc nhẹ", T.C["warn"]),
                    2: ("Nên dừng nghỉ", "#fb923c"), 3: ("Nguy cơ cao", T.C["danger"]),
                }[lv]
                self.fatigue_level.configure(text=text, text_color=color)
                self.fatigue_bar.configure(progress_color=color)
                self.fatigue_bar.set(min(1.0, info["score"] / 8.0))
                if getattr(self, "mini", None) is not None:
                    self.mini.set_fatigue(text, color, info["score"] / 8.0)
                self.fatigue_signs.configure(
                    text="\n".join("• " + s for s in info["signs"][:5])
                    if info["signs"] else "Chưa ghi nhận dấu hiệu bất thường.")
            else:
                self.fatigue_level.configure(text="—", text_color=T.C["muted"])
                self.fatigue_bar.set(0)
                if getattr(self, "mini", None) is not None:
                    self.mini.set_fatigue("—", T.C["muted"], 0)
        except Exception as e:
            log.warning("_tick_slow: %s", e)
        self.root.after(1000, self._tick_slow)

    # ================================================================ PHAT HIEN (giu nguyen tu v3)
    def init_landmarker(self):
        try:
            base = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            return True
        except Exception as e:
            log.error("MediaPipe init failed: %s\n%s", e, traceback.format_exc())
            self.landmarker = None
            return False

    def close_landmarker(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception as e:
                log.warning("close_landmarker: %s", e)
            self.landmarker = None

    def open_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                log.warning("cap.release: %s", e)
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok = self.cap.isOpened()
            if not ok:
                log.error("Camera open failed (isOpened=False)")
            return ok
        except Exception as e:
            log.error("open_camera: %s\n%s", e, traceback.format_exc())
            self.cap = None
            return False

    # ================================================================ DIEU KHIEN
    def _fail_start(self, ui_msg, log_msg):
        self.system_error = True
        self.system_error_msg = ui_msg
        self.status = "SYSTEM ERROR"
        log.error("Start aborted: %s", log_msg)
        self._show_system_error_ui()

    def start(self):
        if self.running:
            return
        self.system_error = False
        self.system_error_msg = ""
        self.detect_restarts = 0
        self.camera_obstructed = False
        self._obstructed_since = None
        self._banner_state = None
        self.close_landmarker()
        if not os.path.exists(MODEL_PATH):
            self._fail_start(
                "Không tìm thấy face_landmarker.task — hãy đặt file model cạnh ứng dụng.",
                "model file missing: %s" % MODEL_PATH)
            return
        if not self.init_landmarker():
            self._fail_start("Không khởi tạo được MediaPipe — xem system_errors.log.",
                             "landmarker init failed")
            return
        if not self.open_camera():
            self.close_landmarker()
            self._fail_start("Không mở được camera — kiểm tra camera đã cắm và không bị ứng dụng khác dùng.",
                             "camera open failed")
            return
        self.running = True
        self.last_heartbeat = time.time()
        self.session_start = time.time()
        self.session_alerts = 0
        self.status = "NORMAL"
        self.reason = ""
        self._set_running_ui(True)
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.advisor.start()
        self.update_gui()
        self._schedule_watchdog()
        log.info("Detection started")

    def stop(self):
        was_running = self.running
        self.running = False
        if not self.system_error:
            self.status = "STOPPED"
            self.reason = ""
        self.env_flag = ""
        self.camera_obstructed = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                log.warning("stop cap.release: %s", e)
            self.cap = None
        self.close_landmarker()
        self.advisor.stop()
        self.health_vision.reset()
        self.session_start = None
        self._set_running_ui(False)
        self._show_stopped_ui()
        if was_running:
            log.info("Detection stopped by user")

    def on_exit(self):
        self.running = False
        try:
            self.trip.stop()          # giu trip_state.json de lan sau tiep tuc
        except Exception:
            pass
        self.advisor.stop()
        try:
            if self.rag_service is not None:
                self.rag_service.close()
                self.rag_service = None
        except Exception:
            pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                log.warning("exit cap.release: %s", e)
        self.close_landmarker()
        log.info("App exit")
        self.root.destroy()

    def _schedule_watchdog(self):
        """Heartbeat: detection im lang qua lau -> SYSTEM ERROR (+ restart co gioi han)."""
        if not self.running:
            return
        try:
            if (
                not self.system_error
                and self.last_heartbeat > 0
                and (time.time() - self.last_heartbeat) > HEARTBEAT_TIMEOUT
            ):
                log.error("Watchdog: no heartbeat for %.1fs", time.time() - self.last_heartbeat)
                if self.detect_restarts < MAX_DETECT_RESTARTS:
                    self.detect_restarts += 1
                    log.warning("Attempting detection restart %d/%d",
                                self.detect_restarts, MAX_DETECT_RESTARTS)
                    self._try_restart_detection()
                else:
                    self.system_error = True
                    self.system_error_msg = "Hệ thống phát hiện đã dừng — không nhận được hình từ camera."
                    self.status = "SYSTEM ERROR"
                    self.reason = ""
                    self.running = False
                    self._show_system_error_ui()
                    log.error("Detection stopped permanently after max restarts")
        except Exception as e:
            log.error("watchdog: %s\n%s", e, traceback.format_exc())

        if self.running and not self.system_error:
            self.root.after(WATCHDOG_CHECK_MS, self._schedule_watchdog)

    def _try_restart_detection(self):
        """Restart nhẹ: đóng/mở lại camera + landmarker, start thread mới."""
        try:
            self.running = False
            time.sleep(0.15)
            self.close_landmarker()
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    log.warning("restart release: %s", e)
                self.cap = None
            if not self.init_landmarker() or not self.open_camera():
                self.system_error = True
                self.system_error_msg = "SYSTEM ERROR - DETECTION STOPPED"
                self.status = "SYSTEM ERROR"
                log.error("Restart failed: init/camera")
                return
            self.running = True
            self.last_heartbeat = time.time()
            self.system_error = False
            self.status = "NORMAL"
            self.thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.thread.start()
            log.info("Detection restarted OK")
        except Exception as e:
            self.system_error = True
            self.system_error_msg = "SYSTEM ERROR - DETECTION STOPPED"
            self.status = "SYSTEM ERROR"
            self.running = False
            log.error("Restart exception: %s\n%s", e, traceback.format_exc())

    def detection_loop(self):
        if self.cap is None or self.landmarker is None:
            self.running = False
            log.error("detection_loop: cap or landmarker is None")
            return

        start_time = time.time()
        prev_time = time.time()
        prev_gray = None
        fail_reads = 0

        eye_closed_start = None
        yawn_start = None
        face_lost_start = None
        is_alarm_on = False
        last_beep_time = 0.0
        last_closed = last_yawn = 0.0
        is_dropped = False
        nod_times = []
        nodding_hold_until = 0.0
        pitch_nod = 0.0
        pitch_nod_ready = False
        last_nod_count_time = 0.0
        drop_since = None
        recover_since = None
        deep_start = None
        deep_duration = 0.0
        was_in_deep_pose = False
        sunglass_streak = 0
        ear_smooth = 0.0
        ear_smooth_ready = False
        open_streak = 0

        alert_active = False
        alert_start_ts = 0.0
        alert_reasons_snap = []

        try:
            while self.running and not self.system_error:
                ret, frame = self.cap.read()
                if not ret:
                    fail_reads += 1
                    if fail_reads >= CAM_FAIL_REOPEN:
                        log.warning("Camera read fail x%d — reconnect", fail_reads)
                        if self.open_camera():
                            fail_reads = 0
                            self.last_heartbeat = time.time()
                        else:
                            time.sleep(0.5)
                    else:
                        time.sleep(0.02)
                    continue
                fail_reads = 0
                self.last_heartbeat = time.time()

                if not self.running or self.landmarker is None:
                    break

                frame = cv2.flip(frame, 1)

                # ----- Camera obstructed (che kín) -----
                if is_frame_obstructed(frame):
                    if self._obstructed_since is None:
                        self._obstructed_since = time.time()
                    elif time.time() - self._obstructed_since >= OBSTRUCTED_HOLD_SEC:
                        self.camera_obstructed = True
                else:
                    self._obstructed_since = None
                    self.camera_obstructed = False

                if self.camera_obstructed:
                    # Không báo buồn ngủ; reset timer detection
                    eye_closed_start = yawn_start = None
                    last_closed = last_yawn = 0.0
                    is_dropped = False
                    drop_since = recover_since = None
                    nod_times = []
                    nodding_hold_until = 0.0
                    deep_start = None
                    deep_duration = 0.0
                    was_in_deep_pose = False
                    open_streak = 0
                    ear_smooth_ready = False
                    if alert_active:
                        log_alert(alert_start_ts, time.time(), alert_reasons_snap)
                        alert_active = False
                    is_alarm_on = False
                    self.status = "CAMERA OBSTRUCTED"
                    self.reason = ""
                    self.env_flag = "obstructed"
                    self.ear = self.mar = self.pitch = self.yaw = 0.0
                    self.nod_count = 0
                    self.closed_dur = self.yawn_dur = self.deep_dur = 0.0
                    self.eye_progress = self.yawn_progress = 0.0
                    self.eye_unreliable = False
                    cv2.putText(
                        frame, "CAMERA OBSTRUCTED", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
                    )
                    rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_show).resize(
                        (640, 480), Image.Resampling.BILINEAR
                    )
                    self.frame_img = img
                    self.frame_seq += 1
                    now = time.time()
                    self.fps = int(1.0 / (now - prev_time + 1e-6))
                    prev_time = now
                    time.sleep(0.01)
                    continue

                self.health_vision.submit(frame)
                frame, light_flag = enhance_lighting(frame)
                h, w = frame.shape[:2]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                high_motion = False
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    if float(np.mean(cv2.absdiff(gray, prev_gray))) > MOTION_DIFF_THRESH:
                        high_motion = True
                prev_gray = gray

                env_parts = []
                if light_flag == "low_light":
                    env_parts.append("low_light")
                elif light_flag == "glare":
                    env_parts.append("glare")
                if high_motion:
                    env_parts.append("motion")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int((time.time() - start_time) * 1000)

                try:
                    result = self.landmarker.detect_for_video(mp_image, ts)
                except ValueError as e:
                    log.warning("detect_for_video ValueError: %s", e)
                    break
                except Exception as e:
                    log.error("detect_for_video: %s\n%s", e, traceback.format_exc())
                    break

                ear_value = mar_value = pitch = yaw = 0.0
                closed_duration = last_closed
                yawn_duration = last_yawn
                nod_count = 0
                is_nodding_alarm = False
                is_deep_alarm = False
                current_th = EAR_THRESHOLD_NORMAL
                eye_unreliable = False

                if result.face_landmarks:
                    face_lost_start = None
                    fl = result.face_landmarks[0]

                    left_ear = eye_aspect_ratio(fl, LEFT_EYE, w, h)
                    right_ear = eye_aspect_ratio(fl, RIGHT_EYE, w, h)
                    mar_value = mouth_aspect_ratio(fl, MOUTH, w, h)
                    pitch = compute_head_pitch_score(fl, w, h)
                    yaw = compute_head_yaw_score(fl, w, h)

                    ratio_th = (
                        SUNGLASS_RATIO_LOWLIGHT
                        if light_flag == "low_light"
                        else SUNGLASS_RATIO_NORMAL
                    )
                    if is_sunglasses_relative(frame, fl, w, h, ratio_th):
                        sunglass_streak += 1
                    else:
                        sunglass_streak = 0
                    eye_unreliable = sunglass_streak >= SUNGLASS_CONFIRM_FRAMES
                    if eye_unreliable:
                        env_parts.append("sunglass")

                    if eyes_overexposed(frame, fl, w, h):
                        eye_unreliable = True
                        if "glare" not in env_parts:
                            env_parts.append("glare")

                    mw = euclidean_distance(
                        (fl[61].x * w, fl[61].y * h),
                        (fl[291].x * w, fl[291].y * h),
                    )
                    fw = euclidean_distance(
                        (fl[33].x * w, fl[33].y * h),
                        (fl[263].x * w, fl[263].y * h),
                    )
                    is_smiling = (mw / (fw + 1e-6)) > SMILE_RATIO_THRESHOLD

                    if yaw > YAW_EXTREME:
                        ear_value = left_ear
                        current_th = EAR_THRESHOLD_EXTREME
                    elif yaw < -YAW_EXTREME:
                        ear_value = right_ear
                        current_th = EAR_THRESHOLD_EXTREME
                    else:
                        ear_value = (left_ear + right_ear) / 2.0
                        if abs(yaw) > YAW_HIGH:
                            current_th = EAR_THRESHOLD_YAW
                        elif pitch > PITCH_HIGH or is_smiling:
                            current_th = EAR_THRESHOLD_STRICT
                        else:
                            current_th = EAR_THRESHOLD_NORMAL

                    if ear_value < EAR_VERY_LOW:
                        current_th = EAR_VERY_LOW

                    if light_flag == "low_light" and not eye_unreliable:
                        if current_th < EAR_THRESHOLD_LOWLIGHT:
                            current_th = EAR_THRESHOLD_LOWLIGHT

                    # EAR mọi frame (kể cả blur)
                    if eye_unreliable:
                        if eye_closed_start is not None:
                            closed_duration = time.time() - eye_closed_start
                        else:
                            closed_duration = 0.0
                    else:
                        if not ear_smooth_ready:
                            ear_smooth = ear_value
                            ear_smooth_ready = True
                        else:
                            ear_smooth = (
                                EAR_SMOOTH_ALPHA * ear_value
                                + (1.0 - EAR_SMOOTH_ALPHA) * ear_smooth
                            )
                        if ear_smooth < current_th:
                            open_streak = 0
                            if eye_closed_start is None:
                                eye_closed_start = time.time()
                            closed_duration = time.time() - eye_closed_start
                        else:
                            open_streak += 1
                            if open_streak >= OPEN_RESET_FRAMES:
                                eye_closed_start = None
                                closed_duration = 0.0
                                open_streak = 0
                            elif eye_closed_start is not None:
                                closed_duration = time.time() - eye_closed_start
                            else:
                                closed_duration = 0.0

                    last_closed = closed_duration

                    # Yawn + Deep: không khóa khi rung
                    if mar_value > MAR_THRESHOLD:
                        if yawn_start is None:
                            yawn_start = time.time()
                        yawn_duration = time.time() - yawn_start
                    else:
                        yawn_start = None
                        yawn_duration = 0.0
                    last_yawn = yawn_duration

                    if pitch > DEEP_PITCH_THRESHOLD:
                        was_in_deep_pose = True
                        if deep_start is None:
                            deep_start = time.time()
                        deep_duration = time.time() - deep_start
                    else:
                        was_in_deep_pose = False
                        deep_start = None
                        deep_duration = 0.0
                    is_deep_alarm = deep_duration >= DEEP_HOLD_TIME

                    # Nod: luôn chạy. Nhiễu/rung → dwell dài hơn (vẫn nhận gật thật)
                    now = time.time()
                    if not pitch_nod_ready:
                        pitch_nod = pitch
                        pitch_nod_ready = True
                    else:
                        pitch_nod = (
                            NOD_PITCH_SMOOTH * pitch
                            + (1.0 - NOD_PITCH_SMOOTH) * pitch_nod
                        )
                    dwell = NOD_DWELL_NOISY if high_motion else NOD_DWELL_NORMAL
                    nod_times = [t for t in nod_times if now - t <= NOD_TIME_WINDOW]

                    if pitch_nod > NOD_DROP_THRESHOLD:
                        recover_since = None
                        if drop_since is None:
                            drop_since = now
                        if (now - drop_since) >= dwell:
                            is_dropped = True
                    elif pitch_nod < NOD_RECOVER_THRESHOLD:
                        drop_since = None
                        if is_dropped:
                            if recover_since is None:
                                recover_since = now
                            if (now - recover_since) >= dwell:
                                if now - last_nod_count_time >= NOD_MIN_GAP:
                                    nod_times.append(now)
                                    last_nod_count_time = now
                                    if len(nod_times) >= NOD_COUNT_NEEDED:
                                        nodding_hold_until = now + NOD_HOLD_TIME
                                        nod_times = []
                                is_dropped = False
                                recover_since = None
                        else:
                            recover_since = None
                    else:
                        recover_since = None

                    nod_count = len(nod_times)
                    is_nodding_alarm = now < nodding_hold_until

                    for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                        lm = fl[idx]
                        cv2.circle(
                            frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 220, 100), -1
                        )
                else:
                    sunglass_streak = 0
                    if face_lost_start is None:
                        face_lost_start = time.time()
                    lost = time.time() - face_lost_start

                    if was_in_deep_pose:
                        if deep_start is None:
                            deep_start = time.time() - deep_duration
                        deep_duration = time.time() - deep_start
                        is_deep_alarm = deep_duration >= DEEP_HOLD_TIME
                        if lost >= DEEP_NO_FACE_GRACE:
                            eye_closed_start = yawn_start = None
                            last_closed = last_yawn = closed_duration = yawn_duration = 0.0
                            is_dropped = False
                            drop_since = recover_since = None
                            nod_times = []
                            nodding_hold_until = 0.0
                            deep_start = None
                            deep_duration = 0.0
                            was_in_deep_pose = False
                            is_deep_alarm = False
                            is_alarm_on = False
                            open_streak = 0
                            ear_smooth_ready = False
                    else:
                        is_deep_alarm = False
                        if lost >= FACE_LOST_GRACE:
                            eye_closed_start = yawn_start = None
                            last_closed = last_yawn = closed_duration = yawn_duration = 0.0
                            is_dropped = False
                            drop_since = recover_since = None
                            nod_times = []
                            nodding_hold_until = 0.0
                            deep_start = None
                            deep_duration = 0.0
                            was_in_deep_pose = False
                            is_alarm_on = False
                            open_streak = 0
                            ear_smooth_ready = False

                eye_alarm = (not eye_unreliable) and (
                    closed_duration >= EYE_CLOSED_TIME_LIMIT
                )
                yawn_alarm = yawn_duration >= YAWN_TIME_LIMIT
                should_alarm = (
                    eye_alarm or yawn_alarm or is_nodding_alarm or is_deep_alarm
                )

                if should_alarm:
                    reasons = []
                    if eye_alarm:
                        reasons.append("Mắt nhắm")
                    if yawn_alarm:
                        reasons.append("Ngáp")
                    if is_nodding_alarm:
                        reasons.append("Gật gù")
                    if is_deep_alarm:
                        reasons.append("Cúi sâu")

                    if not is_alarm_on:
                        is_alarm_on = True
                        alert_active = True
                        alert_start_ts = time.time()
                        alert_reasons_snap = list(reasons)
                        self.session_alerts += 1
                        for _r in reasons:
                            self.advisor.record_event(_r)
                        beep_async(1000, 160)
                        last_beep_time = time.time()
                    else:
                        alert_reasons_snap = list(reasons)
                        if time.time() - last_beep_time >= BEEP_INTERVAL:
                            beep_async(1000, 120)
                            last_beep_time = time.time()
                    self.reason = " + ".join(reasons)
                    self.status = "WARNING"
                else:
                    if alert_active:
                        log_alert(alert_start_ts, time.time(), alert_reasons_snap)
                        alert_active = False
                    is_alarm_on = False
                    self.reason = ""
                    self.status = "NORMAL"

                self.ear, self.mar = ear_value, mar_value
                self.pitch, self.yaw = pitch, yaw
                self.nod_count = nod_count
                self.closed_dur, self.yawn_dur = closed_duration, yawn_duration
                self.deep_dur = deep_duration
                self.eye_progress = min(
                    100.0, closed_duration / EYE_CLOSED_TIME_LIMIT * 100
                )
                self.yawn_progress = min(100.0, yawn_duration / YAWN_TIME_LIMIT * 100)
                self.eye_unreliable = eye_unreliable
                self.current_th_display = current_th
                self.env_flag = ", ".join(env_parts)

                now = time.time()
                self.fps = int(1.0 / (now - prev_time + 1e-6))
                prev_time = now

                if eye_unreliable and "sunglass" in self.env_flag:
                    cv2.putText(
                        frame, "EAR skipped (sunglass)", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                    )
                elif eye_unreliable:
                    cv2.putText(
                        frame, "EAR skipped (glare/eyes)", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                    )
                elif light_flag == "low_light":
                    cv2.putText(
                        frame, f"Low-light EAR th={current_th:.2f}", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                    )
                if high_motion:
                    cv2.putText(
                        frame, "Motion: stricter nod dwell", (12, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                    )

                rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_show).resize(
                    (640, 480), Image.Resampling.BILINEAR
                )
                self.frame_img = img
                self.frame_seq += 1
                time.sleep(0.006)

        except Exception as e:
            log.error("detection_loop crashed: %s\n%s", e, traceback.format_exc())
            self.system_error = True
            self.system_error_msg = "SYSTEM ERROR - DETECTION FAILED"
            self.status = "SYSTEM ERROR"
            self.running = False
            # Cập nhật UI ngay trên main thread
            try:
                self.ui.post(self._show_system_error_ui)
            except Exception as ui_e:
                log.warning("Could not schedule SYSTEM ERROR UI: %s", ui_e)

        if alert_active:
            log_alert(alert_start_ts, time.time(), alert_reasons_snap)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                log.warning("loop end release: %s", e)
            self.cap = None

    # ================================================================ CAP NHAT GIAO DIEN
    def _show_system_error_ui(self):
        """Goi tu main thread khi he thong loi."""
        try:
            msg = self.system_error_msg or "Hệ thống phát hiện gặp lỗi."
            self._pill("LỖI HỆ THỐNG", "danger")
            self._set_banner(msg, "danger")
            self._set_running_ui(False)
            for t in self.tiles.values():
                t.set("—")
        except Exception as e:
            log.warning("_show_system_error_ui: %s", e)

    def _show_stopped_ui(self):
        try:
            if self.system_error:
                return
            self._pill("ĐÃ DỪNG", "idle")
            self._set_banner("Nhấn «Bắt đầu giám sát» để khởi động camera.", "idle")
            for t in self.tiles.values():
                t.set("—")
            self.eye_bar.set(0)
            self.yawn_bar.set(0)
            self.tile_fps.set("— FPS")
            self.frame_img = None
            self._shown_seq = -1
            self.photo = None
            self.video_label.configure(image="", text="Camera đang tắt\nNhấn «Bắt đầu giám sát»")
            self.video_label.image = None
            if getattr(self, "mini", None) is not None:
                self.mini.clear_frame()
        except Exception as e:
            log.warning("_show_stopped_ui: %s", e)

    def _refresh_photo(self):
        """Main thread: PIL -> PhotoImage (Tkinter khong an toan da luong)."""
        img = self.frame_img
        if img is None or self.frame_seq == self._shown_seq:
            return
        self._shown_seq = self.frame_seq
        try:
            if img.size != self._video_size:
                img = img.resize(self._video_size, Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=self.photo, text="")
            self.video_label.image = self.photo
            mini = getattr(self, "mini", None)
            if mini is not None and self.current_page == "map" and self.frame_seq % 3 == 0:
                from mini_monitor import VIDEO_W, VIDEO_H
                mini.set_frame(ImageTk.PhotoImage(
                    image=self.frame_img.resize((VIDEO_W, VIDEO_H), Image.Resampling.BILINEAR)))
        except Exception as e:
            log.warning("PhotoImage: %s", e)

    def update_gui(self):
        self._refresh_photo()

        if self.system_error:
            self._show_system_error_ui()
            return

        if not self.running and self.status == "STOPPED":
            return

        if self.camera_obstructed and self.running:
            self._pill("CAMERA BỊ CHE", "warn")
            self._set_banner("Camera bị che hoặc không có hình — hãy kiểm tra ống kính.", "warn")
            for t in self.tiles.values():
                t.set("—")
            self.tile_fps.set(f"{self.fps} FPS")
            self.eye_bar.set(0)
            self.yawn_bar.set(0)
            self.root.after(30, self.update_gui)
            return

        if self.status == "WARNING":
            self._pill("CẢNH BÁO", "danger", self.reason or "")
            self._set_banner("CẢNH BÁO  ·  " + (self.reason or ""), "danger")
        else:
            self._pill("BÌNH THƯỜNG", "ok")
            if time.time() < self._thanks_until:
                self._set_banner("Cảm ơn bạn, hãy chú ý an toàn khi lái xe.", "ok")
            elif self.env_flag:
                self._set_banner("Đang hoạt động  ·  điều kiện: " + self.env_flag, "idle")
            else:
                self._set_banner("Hệ thống hoạt động bình thường — chưa phát hiện dấu hiệu bất thường.",
                                 "idle")

        if self.eye_unreliable:
            self.tiles["ear"].set(f"{self.ear:.3f} ·bỏ qua")
        else:
            self.tiles["ear"].set(f"{self.ear:.3f}")
        self.tiles["mar"].set(f"{self.mar:.3f}")
        self.tiles["pitch"].set(f"{self.pitch:.2f}")
        self.tiles["yaw"].set(f"{self.yaw:.2f}")
        self.tiles["nod"].set(str(self.nod_count))
        self.tiles["deep"].set(f"{self.deep_dur:.1f}s")
        self.tiles["closed"].set(f"{self.closed_dur:.1f}s")
        self.tiles["yawn"].set(f"{self.yawn_dur:.1f}s")
        self.tile_fps.set(f"{self.fps} FPS")
        self.eye_bar.set(self.eye_progress / 100.0)
        self.yawn_bar.set(self.yawn_progress / 100.0)

        if self.running:
            self.root.after(30, self.update_gui)


if __name__ == "__main__":
    T.setup()
    root = ctk.CTk()
    app = DrowsinessApp(root)
    root.mainloop()