"""
trip_session.py - Theo dõi hành trình THỜI GIAN THỰC theo kế hoạch (trip_planner).

TripSession (logic thuần, test được):
  update(lat, lon, now, accuracy_m) -> trạng thái + sự kiện:
    approaching_stop / skipped_stop / overdue_rest / hard_limit / off_route / arrived
  Theo dõi: tiến độ dọc tuyến, còn bao nhiêu km, điểm dừng kế tiếp, thời gian lái LIÊN TỤC
  (tự reset khi xe dừng >= 10 phút), tổng thời gian lái, điểm dừng đã ghé.
  Nếu chỉ có vị trí IP (sai số km) -> chuyển sang ước tính theo thời gian và nói rõ độ tin cậy thấp.

TripSupervisor: thread nền, hỏi vị trí định kỳ, đẩy sự kiện vào hàng đợi cho GUI,
  tự lập lại kế hoạch khi lệch tuyến hoặc nhắc nghỉ quá hạn, lưu trạng thái ra đĩa.
"""
import os
import json
import time
import queue
import logging
import threading
import datetime as dt

import trip_planner as tp
from geo_utils import haversine_m

log = logging.getLogger("drowsiness")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(BASE_DIR, "trip_state.json")

MOVING_MPS = 1.5                 # >= ~5,4 km/h coi là đang chạy
REST_RESET_MIN = 10              # dừng >= 10 phút = đã nghỉ, reset đồng hồ lái liên tục
VISIT_RADIUS_M = 400
OFF_ROUTE_M = 300
OFF_ROUTE_FIXES = 3              # lệch tuyến liên tiếp bao nhiêu lần mới tính
LOW_ACCURACY_M = 1000            # sai số lớn hơn: không dùng để đo di chuyển/lệch tuyến
APPROACH_M = 5000
SKIP_MARGIN_M = 2000
OVERDUE_EXTRA_MIN = 15
MAX_DT_S = 120                   # một lần cập nhật cộng tối đa 120 s (tránh máy ngủ làm sai số)
EVENT_COOLDOWN_MIN = {"overdue_rest": 20, "hard_limit": 10, "off_route": 5}


class TripSession:
    def __init__(self, route, plan, origin=None, destination=None, now=None,
                 rest_every_min=None, max_continuous_min=None):
        self.rest_every = float(rest_every_min or tp.REST_EVERY_MIN)
        self.max_cont = float(max_continuous_min or tp.MAX_CONTINUOUS_MIN)
        self.origin, self.destination = origin, destination
        self.cont_drive_min = float(plan.get("assumptions", {}).get("already_driven_min", 0) or 0)
        self.total_drive_min = self.cont_drive_min
        self.fatigue_until = None
        self.visited = set()
        self.events_fired = {}
        self.off_count = 0
        self.last_fix = None                 # (now, lat, lon)
        self.stationary_since = None
        self.progress_m = 0.0
        self.mode = "gps"
        self.load(route, plan, now)

    # ---- nạp tuyến / kế hoạch (cũng dùng khi lập lại) ----
    def load(self, route, plan, now=None):
        self.route, self.plan = route, plan
        self.idx = tp.RouteIndex(route["points"])
        self.stops = plan["stops"]
        self.visited = set()
        self.progress_m = 0.0
        self.off_count = 0
        self.started = now or dt.datetime.now()
        self.min_per_m = (float(route["duration"]) / 60.0 * tp.TIME_FACTOR) / max(self.idx.total, 1.0)

    # ---- ước tính theo thời gian (khi không có GPS tốt) ----
    def _progress_by_time(self, now):
        sched = [(self.plan["depart"], 0.0)]
        for s in self.stops:
            sched.append((s["eta"], s["at_m"]))
            sched.append((s["depart_after"], s["at_m"]))
        sched.append((self.plan["arrival"], self.idx.total))
        if now <= sched[0][0]:
            return 0.0
        for (t0, m0), (t1, m1) in zip(sched, sched[1:]):
            if t0 <= now <= t1:
                span = (t1 - t0).total_seconds() or 1.0
                return m0 + (m1 - m0) * (now - t0).total_seconds() / span
        return self.idx.total

    def _project(self, lat, lon):
        """Chiếu vị trí lên tuyến, ưu tiên đoạn gần tiến độ cũ (tránh nhảy nhầm đoạn khi tuyến quay lại)."""
        best = None
        lo, hi = self.progress_m - 1000, self.progress_m + 40000
        for i, (plat, plon) in enumerate(self.idx.pts):
            if not (lo <= self.idx.cum[i] <= hi):
                continue
            d = haversine_m(lat, lon, plat, plon)
            if best is None or d < best[1]:
                best = (self.idx.cum[i], d)
        if best is None or best[1] > OFF_ROUTE_M * 2:
            return self.idx.project(lat, lon)         # tìm toàn tuyến
        return best

    def _cooldown_ok(self, key, now):
        last = self.events_fired.get(key)
        mins = EVENT_COOLDOWN_MIN.get(key, 10 ** 6)
        if last is None or (now - last).total_seconds() / 60.0 >= mins:
            self.events_fired[key] = now
            return True
        return False

    def tighten(self, now, minutes=30):
        """Có dấu hiệu mệt mỏi: ngưỡng nhắc nghỉ chặt hơn trong một lúc."""
        self.fatigue_until = now + dt.timedelta(minutes=minutes)

    # ---- cập nhật chính ----
    def update(self, lat, lon, now, accuracy_m=None):
        ev = []
        low = accuracy_m is None or accuracy_m > LOW_ACCURACY_M
        self.mode = "time" if low else "gps"
        off_m = 0.0

        if low:
            self.progress_m = max(self.progress_m, self._progress_by_time(now))
        else:
            prog, off_m = self._project(lat, lon)
            moving = False
            if self.last_fix:
                t0, la0, lo0 = self.last_fix
                dts = min((now - t0).total_seconds(), MAX_DT_S)
                if dts > 0:
                    speed = haversine_m(la0, lo0, lat, lon) / max((now - t0).total_seconds(), 1e-6)
                    moving = speed >= MOVING_MPS
                    if moving:
                        self.cont_drive_min += dts / 60.0
                        self.total_drive_min += dts / 60.0
                        self.stationary_since = None
                    elif self.stationary_since is None:
                        self.stationary_since = t0
            self.last_fix = (now, lat, lon)
            if off_m <= OFF_ROUTE_M * 2:
                self.progress_m = max(self.progress_m, prog)
            # nghỉ đủ lâu -> reset đồng hồ lái liên tục + ghi nhận điểm dừng đã ghé
            if self.stationary_since is not None:
                rest_min = (now - self.stationary_since).total_seconds() / 60.0
                if rest_min >= 3:
                    for i, s in enumerate(self.stops):
                        if haversine_m(lat, lon, s["lat"], s["lon"]) <= VISIT_RADIUS_M:
                            self.visited.add(i)
                if rest_min >= REST_RESET_MIN and self.cont_drive_min > 0:
                    self.cont_drive_min = 0.0
                    ev.append({"type": "rested", "level": 0})
            # lệch tuyến
            self.off_count = self.off_count + 1 if off_m > OFF_ROUTE_M else 0
            if self.off_count >= OFF_ROUTE_FIXES and self._cooldown_ok("off_route", now):
                ev.append({"type": "off_route", "level": 1, "off_m": off_m, "needs_replan": True})

        remaining_m = max(0.0, self.idx.total - self.progress_m)
        pending = [(i, s) for i, s in enumerate(self.stops) if i not in self.visited]

        # điểm dừng bị bỏ qua
        for i, s in pending:
            if self.progress_m > s["at_m"] + SKIP_MARGIN_M and self.mode == "gps":
                key = f"skip{i}"
                if key not in self.events_fired:
                    self.events_fired[key] = now
                    self.visited.add(i)
                    ev.append({"type": "skipped_stop", "level": 1, "stop": s})
        pending = [(i, s) for i, s in enumerate(self.stops)
                   if i not in self.visited and s["at_m"] >= self.progress_m - SKIP_MARGIN_M]

        nxt = None
        if pending:
            i, s = pending[0]
            ahead = max(0.0, s["at_m"] - self.progress_m)
            nxt = {"index": i, "name": s["name"], "kind": s["kind"], "lat": s["lat"], "lon": s["lon"],
                   "dist_m": ahead, "eta_min": ahead * self.min_per_m, "known": s.get("known", True),
                   "purpose": s.get("purpose", "rest")}
            tight = self.fatigue_until is not None and now < self.fatigue_until
            radius = APPROACH_M * (2 if tight else 1)
            if ahead <= radius and f"appr{i}" not in self.events_fired:
                self.events_fired[f"appr{i}"] = now
                ev.append({"type": "approaching_stop", "level": 0, "stop": s, "dist_m": ahead})

        # nhắc nghỉ quá hạn / vượt mức lái liên tục
        if self.mode == "gps":
            tight = self.fatigue_until is not None and now < self.fatigue_until
            limit = self.rest_every + OVERDUE_EXTRA_MIN - (30 if tight else 0)
            soon = nxt is not None and nxt["eta_min"] <= 20
            if self.cont_drive_min >= self.max_cont and self._cooldown_ok("hard_limit", now):
                ev.append({"type": "hard_limit", "level": 3})
            elif self.cont_drive_min >= limit and not soon and self._cooldown_ok("overdue_rest", now):
                ev.append({"type": "overdue_rest", "level": 2, "needs_replan": nxt is None})

        arrived = remaining_m <= 300
        if arrived and "arrived" not in self.events_fired:
            self.events_fired["arrived"] = now
            ev.append({"type": "arrived", "level": 0})

        eta = now + dt.timedelta(minutes=remaining_m * self.min_per_m
                                 + sum(s["stay_min"] for _, s in pending))
        return {"mode": self.mode, "progress_km": self.progress_m / 1000.0,
                "remaining_km": remaining_m / 1000.0, "off_route_m": off_m,
                "eta": eta, "next_stop": nxt, "cont_drive_min": self.cont_drive_min,
                "total_drive_min": self.total_drive_min, "events": ev,
                "arrived": arrived, "visited": len(self.visited)}

    # ---- lưu / khôi phục ----
    def to_json(self):
        r = self.route
        return {"route": {"points": [[round(a, 5), round(b, 5)] for a, b in r["points"]],
                          "distance": r["distance"], "duration": r["duration"]},
                "plan": tp.plan_to_json(self.plan), "origin": self.origin, "destination": self.destination,
                "visited": sorted(self.visited), "cont_drive_min": self.cont_drive_min,
                "total_drive_min": self.total_drive_min, "progress_m": self.progress_m,
                "saved_at": dt.datetime.now().isoformat()}

    @classmethod
    def from_json(cls, d):
        route = {"points": [tuple(p) for p in d["route"]["points"]],
                 "distance": d["route"]["distance"], "duration": d["route"]["duration"]}
        s = cls(route, tp.plan_from_json(d["plan"]), d.get("origin"), d.get("destination"))
        s.visited = set(d.get("visited", []))
        s.cont_drive_min = d.get("cont_drive_min", 0.0)
        s.total_drive_min = d.get("total_drive_min", 0.0)
        s.progress_m = d.get("progress_m", 0.0)
        return s


def save_state(session, path=STATE_PATH):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(session.to_json(), f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as exc:
        log.info("trip_state: không lưu được: %s", exc)


def load_state(path=STATE_PATH, max_age_h=12):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        age = dt.datetime.now() - dt.datetime.fromisoformat(d["saved_at"])
        if age > dt.timedelta(hours=max_age_h):
            return None
        return TripSession.from_json(d)
    except Exception:
        return None


def clear_state(path=STATE_PATH):
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------- Văn bản sự kiện (do code viết) ----------------
def event_text(ev, status):
    t = ev["type"]
    if t == "approaching_stop":
        s = ev["stop"]
        return (f"Sắp tới điểm dừng đã lên kế hoạch: {s['name']}, còn khoảng "
                f"{tp.fmt_km(ev['dist_m'])}. Hãy dừng nghỉ ở đó.")
    if t == "skipped_stop":
        return f"Bạn vừa đi qua điểm dừng «{ev['stop']['name']}» mà chưa dừng. Hãy chọn điểm nghỉ kế tiếp."
    if t == "overdue_rest":
        return (f"Bạn đã lái liên tục khoảng {tp.fmt_dur(status['cont_drive_min'])}. "
                "Hãy dừng nghỉ ở nơi an toàn gần nhất.")
    if t == "hard_limit":
        return (f"Bạn đã lái liên tục {tp.fmt_dur(status['cont_drive_min'])}, quá mức khuyến nghị. "
                "Cần dừng xe nghỉ ngay khi có thể làm an toàn.")
    if t == "off_route":
        return "Bạn đang lệch khỏi tuyến đã lên kế hoạch. Hệ thống sẽ tính lại lộ trình."
    if t == "arrived":
        return "Bạn đã đến nơi. Chúc bạn an toàn."
    if t == "rested":
        return "Đã ghi nhận bạn nghỉ đủ lâu. Đồng hồ lái liên tục được đặt lại."
    if t == "replanned":
        return "Đã lập lại kế hoạch từ vị trí hiện tại."
    return ""


class TripSupervisor:
    """Thread nền: hỏi vị trí -> cập nhật session -> đẩy sự kiện vào hàng đợi (GUI tự poll)."""

    def __init__(self, get_fix, replan_fn, interval_s=10.0, now_fn=dt.datetime.now):
        self.get_fix = get_fix            # -> {"lat","lon","accuracy_m"} hoặc None
        self.replan_fn = replan_fn        # (session, fix, status) -> (route, plan) hoặc None
        self.interval = interval_s
        self.now_fn = now_fn
        self.session = None
        self.status = None
        self.events = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self, session):
        self.stop()
        with self._lock:
            self.session, self.status = session, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True)
        self._thread.start()

    def stop(self, clear=False):
        self._stop.set()
        if clear:
            with self._lock:
                self.session, self.status = None, None
            clear_state()

    @property
    def active(self):
        return self.session is not None and not self._stop.is_set()

    def notify_fatigue(self, level):
        s = self.session
        if s is not None and level >= 2:
            s.tighten(self.now_fn())

    def step(self):
        """Một vòng cập nhật (tách riêng để test)."""
        s = self.session
        if s is None:
            return None
        fix = self.get_fix()
        if not fix:
            return self.status
        now = self.now_fn()
        with self._lock:
            st = s.update(fix["lat"], fix["lon"], now, fix.get("accuracy_m"))
            self.status = st
        for ev in st["events"]:
            if ev.get("needs_replan"):
                try:
                    out = self.replan_fn(s, fix, st)
                    if out:
                        route, plan = out
                        with self._lock:
                            s.cont_drive_min = float(plan.get("assumptions", {}).get("already_driven_min", 0))
                            s.load(route, plan, now)
                        self.events.put({"type": "replanned", "level": 0, "text": event_text({"type": "replanned"}, st)})
                except Exception as exc:
                    log.warning("Trip replan lỗi: %s", exc)
            self.events.put(dict(ev, text=event_text(ev, st), position=fix))
        if st["arrived"]:
            self.stop(clear=True)
        else:
            save_state(s)
        return st

    def _run(self, stop):
        while not stop.is_set():
            try:
                self.step()
            except Exception as exc:
                log.warning("TripSupervisor: %s", exc)
            stop.wait(self.interval)
