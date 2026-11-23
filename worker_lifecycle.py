"""
worker_lifecycle.py - Vong doi worker + Supervisor (dung/restart CO XAC NHAN, khong chan UI).

Trang thai: STARTING -> RUNNING -> STOPPING -> STOPPED ; moi trang thai co the -> FAILED.
Chuyen sai bi tu choi + log ILLEGAL_TRANSITION.
Supervisor.tick() la state machine thuan: khong sleep, khong join dai; chi goi is_alive().
Chi goi start()/stop()/tick() tu MAIN thread.
"""
import dataclasses
import enum
import itertools
import logging
import sys
import threading
import time
import traceback

log = logging.getLogger("drowsiness")


class WorkerState(enum.Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


S = WorkerState
_VALID = {
    S.STARTING: {S.RUNNING, S.STOPPING, S.FAILED},
    S.RUNNING: {S.STOPPING, S.FAILED},
    S.STOPPING: {S.STOPPED, S.FAILED},
    S.STOPPED: set(),
    S.FAILED: set(),
}


class WorkerHandle:
    def __init__(self, worker_id, clock=time.monotonic, heartbeat_timeout=2.5,
                 open_grace=10.0, is_restart=False):
        self.id = worker_id
        self.tid = None
        self.thread = None
        self.is_restart = is_restart
        self.stop_event = threading.Event()
        self.lease = None
        self.state = S.STARTING
        self.stop_reason = ""
        self.fail_reason = ""
        self.fail_msg = ""
        self.restartable = True
        self.stop_requested_at = None
        self.running_since = None
        self._clock = clock
        self._timeout = heartbeat_timeout
        self._lock = threading.RLock()
        self._cleanups = []
        self.created = clock()
        self.deadline = self.created + open_grace       # han heartbeat

    # ---- API cho worker ----
    def beat(self, grace=None):
        """Cap nhat han heartbeat. grace=... khi sap lam viec chan lau (mo camera)."""
        self.deadline = self._clock() + (self._timeout if grace is None else grace)

    def should_stop(self):
        if self.stop_event.is_set():
            return True
        lease = self.lease
        return lease is not None and not lease.active    # mat quyen so huu => thoat

    def mark_running(self, reason="init_ok"):
        ok = self._transition(S.RUNNING, reason)
        if ok:
            self.beat()
        return ok

    def fail(self, reason, restartable=True, msg=""):
        """Worker bao loi (lan dau thang). Vong lap thoat, finish() se chuyen sang FAILED."""
        with self._lock:
            if self.fail_reason:
                return False
            self.fail_reason, self.restartable, self.fail_msg = reason, restartable, msg
            if self.stop_requested_at is None:
                self.stop_requested_at = self._clock()
        self.stop_event.set()
        log.error("WORKER_FAIL id=%s tid=%s state=%s reason=%s restartable=%s",
                  self.id, self.tid, self.state.value, reason, restartable)
        return True

    def add_cleanup(self, fn, name=""):
        """Dang ky don dep; chay NGUOC thu tu dang ky trong finally cua worker."""
        self._cleanups.append((name or getattr(fn, "__name__", "fn"), fn))

    # ---- API cho supervisor ----
    def request_stop(self, reason):
        with self._lock:
            if self.state not in (S.STARTING, S.RUNNING):
                return False
            self.stop_reason = reason
            if self.stop_requested_at is None:
                self.stop_requested_at = self._clock()
            self._transition(S.STOPPING, reason)
        self.stop_event.set()
        log.info("STOP_REQUEST id=%s tid=%s reason=%s", self.id, self.tid, reason)
        return True

    def run_cleanups(self):
        errors = 0
        for name, fn in reversed(self._cleanups):
            try:
                fn()
            except Exception as exc:
                errors += 1
                log.error("WORKER_CLEANUP_ERROR id=%s step=%s: %s", self.id, name, exc)
        self._cleanups.clear()
        return errors

    def finish(self, cleanup_errors=0):
        """Goi SAU run_cleanups: chuyen sang trang thai cuoi."""
        with self._lock:
            if self.state in (S.STOPPED, S.FAILED):
                return
            if cleanup_errors and not self.fail_reason:
                self.fail_reason = "cleanup_error"
            if self.fail_reason:
                self._transition(S.FAILED, self.fail_reason)
            elif self.state is S.STOPPING:
                self._transition(S.STOPPED, self.stop_reason or "stopped")
            else:                      # STARTING/RUNNING ma thoat khong co ly do
                self.fail_reason = "exited_without_request"
                self.fail_msg = "SYSTEM ERROR - DETECTION STOPPED"
                self._transition(S.FAILED, self.fail_reason)

    def _transition(self, new, reason):
        with self._lock:
            old = self.state
            if new not in _VALID[old]:
                log.warning("ILLEGAL_TRANSITION id=%s tid=%s %s->%s reason=%s",
                            self.id, self.tid, old.value, new.value, reason)
                return False
            self.state = new
            if new is S.RUNNING:
                self.running_since = self._clock()
            log.info("WORKER_LIFECYCLE id=%s tid=%s %s->%s reason=%s elapsed=%.2fs",
                     self.id, self.tid, old.value, new.value, reason,
                     self._clock() - self.created)
            return True


@dataclasses.dataclass
class SupervisorConfig:
    heartbeat_timeout: float = 2.5
    open_grace: float = 10.0
    stop_join_timeout: float = 3.0
    restart_settle: float = 0.3
    max_restarts: int = 2
    stable_reset: float = 120.0


MSG_NO_FRAME = "Hệ thống phát hiện đã dừng — không nhận được hình từ camera."
MSG_STOPPED = "SYSTEM ERROR - DETECTION STOPPED"
MSG_ZOMBIE = "Không dừng được luồng camera — hãy khởi động lại ứng dụng."


class Supervisor:
    IDLE, RUN, STOPPING, RESTART_WAIT, ERROR, ZOMBIE = (
        "IDLE", "RUN", "STOPPING", "RESTART_WAIT", "ERROR", "ZOMBIE")

    def __init__(self, camera, worker_main, on_event=None, config=None, clock=time.monotonic):
        self.camera = camera
        self._worker_main = worker_main          # callable(handle) chay tren worker thread
        self._on_event = on_event                # callable(kind, info) - chay trong tick (main)
        self.cfg = config or SupervisorConfig()
        self._clock = clock
        self._ids = itertools.count(1)
        self.worker = None
        self.phase = self.IDLE
        self.intent = None                       # user | exit | restart | error
        self.restarts = 0
        self._pending_start = False
        self._restart_at = 0.0
        self._stopping_since = 0.0
        self._intent_msg = ""
        self._reported_running = None

    # ---------------- truy van ----------------
    @staticmethod
    def _alive(h):
        return h is not None and h.thread is not None and h.thread.is_alive()

    @property
    def is_active(self):
        """Y dinh 'dang chay' cho UI (self.running)."""
        if self._pending_start or self.phase in (self.RUN, self.RESTART_WAIT):
            return True
        return self.phase == self.STOPPING and self.intent == "restart"

    def is_idle(self):
        return (not self._alive(self.worker)) and self.camera.owner is None

    def _emit(self, kind, **info):
        if self._on_event:
            try:
                self._on_event(kind, info)
            except Exception as exc:
                log.error("Supervisor event %s loi: %s\n%s", kind, exc, traceback.format_exc())

    # ---------------- lenh ----------------
    def start(self):
        """True neu da nhan lenh (spawn ngay hoac xep hang sau khi dung xong)."""
        if self.phase == self.STOPPING:
            if self.intent == "user":
                self._pending_start = True
                log.info("START queued: cho worker %s dung xong", self.worker.id)
                return True
            return self.intent == "restart"
        if self.phase in (self.RUN, self.RESTART_WAIT):
            return True
        if self.phase == self.ZOMBIE:
            if self._alive(self.worker):
                log.error("START refused: worker %s con song (zombie)", self.worker.id)
                return False
            log.warning("ZOMBIE_RECOVERED id=%s", self.worker.id)
            self.phase = self.IDLE
        self.restarts = 0
        return self._spawn("user_start")

    def stop(self, reason="user"):
        self._pending_start = False
        intent = "exit" if reason == "exit" else "user"
        now = self._clock()
        if self.phase == self.RUN:
            self._enter_stopping(intent, reason, self.worker, now)
        elif self.phase == self.STOPPING:
            if self.intent == "restart" or (intent == "exit" and self.intent == "user"):
                self.intent = intent
        elif self.phase == self.RESTART_WAIT:       # worker cu da chet, chua tao moi
            self.phase = self.IDLE
            self._emit("stopped", reason=reason)

    # ---------------- vong lap ----------------
    def tick(self):
        now = self._clock()
        if self.phase == self.RUN:
            self._tick_run(now)
        if self.phase == self.STOPPING:
            self._tick_stopping(now)
        elif self.phase == self.RESTART_WAIT:
            if now >= self._restart_at:
                self._do_restart()
        elif self.phase == self.ZOMBIE:
            if not self._alive(self.worker):
                log.warning("ZOMBIE_RECOVERED id=%s (thread da thoat, can bam Bat dau de chay lai)",
                            self.worker.id)
                self.phase = self.ERROR

    def _tick_run(self, now):
        h = self.worker
        if h.state is S.RUNNING and self._reported_running != h.id:
            self._reported_running = h.id
            self._emit("running", worker_id=h.id, restart=h.is_restart)
        if h.state in (S.STOPPED, S.FAILED):
            self._handle_failure(h, h.fail_reason or h.stop_reason or "ended_unexpectedly",
                                 h.restartable, h.fail_msg, now)
            return
        if h.stop_event.is_set():                   # dang tu dung (fail) nhung don dep chua xong
            if now - (h.stop_requested_at or now) > self.cfg.stop_join_timeout:
                self._declare_zombie(h, now)
            return
        if (h.state is S.RUNNING and self.restarts and h.running_since is not None
                and now - h.running_since >= self.cfg.stable_reset):
            log.info("RESTART_COUNTER_RESET id=%s was=%d stable=%.0fs",
                     h.id, self.restarts, now - h.running_since)
            self.restarts = 0
        if now > h.deadline:
            reason = "startup_timeout" if h.state is S.STARTING else "heartbeat_timeout"
            log.error("WATCHDOG id=%s tid=%s state=%s reason=%s silent=%.1fs",
                      h.id, h.tid, h.state.value, reason, now - (h.deadline - self.cfg.heartbeat_timeout))
            self._handle_failure(h, reason, True, MSG_NO_FRAME, now)

    def _tick_stopping(self, now):
        h = self.worker
        if not self._alive(h):
            self._finalize_stop(h, now)
        elif now - self._stopping_since > self.cfg.stop_join_timeout:
            self._declare_zombie(h, now)

    # ---------------- chinh sach ----------------
    def _handle_failure(self, h, reason, restartable, msg, now):
        if restartable and self.restarts < self.cfg.max_restarts:
            self.restarts += 1
            log.warning("RESTART_PLAN id=%s n=%d/%d reason=%s",
                        h.id, self.restarts, self.cfg.max_restarts, reason)
            self._emit("restarting", n=self.restarts, max=self.cfg.max_restarts, reason=reason)
            self._enter_stopping("restart", reason, h, now)
        else:
            self._enter_stopping("error", reason, h, now, msg or MSG_STOPPED)

    def _enter_stopping(self, intent, reason, h, now, msg=""):
        self.phase = self.STOPPING
        self.intent = intent
        self._intent_msg = msg
        self._stopping_since = now
        h.request_stop(reason)                       # no-op neu da o trang thai cuoi

    def _finalize_stop(self, h, now):
        """Worker da thread-dead: kiem tra owner=None roi moi di tiep."""
        if self.camera.owner is not None:
            log.critical("INVARIANT_VIOLATION camera owner=%s sau khi worker %s da chet",
                         self.camera.owner, h.id)
        log.info("STOP_CONFIRMED id=%s state=%s intent=%s waited=%.2fs",
                 h.id, h.state.value, self.intent, now - self._stopping_since)
        intent = self.intent
        if intent in ("user", "exit"):
            self.phase = self.IDLE
            self._emit("stopped", reason=intent)
            if self._pending_start:
                self._pending_start = False
                self.restarts = 0
                if not self._spawn("queued_start"):
                    self._set_error(h, "spawn_refused", MSG_STOPPED)
        elif intent == "restart":
            self.phase = self.RESTART_WAIT
            self._restart_at = now + self.cfg.restart_settle
        else:
            self._set_error(h, h.fail_reason or h.stop_reason or "error", self._intent_msg)

    def _set_error(self, h, reason, message, zombie=False):
        self.phase = self.ERROR if not zombie else self.ZOMBIE
        self._emit("error", reason=reason, message=message or MSG_STOPPED,
                   during_start=(h.running_since is None), zombie=zombie, worker_id=h.id)

    def _declare_zombie(self, h, now):
        frame = sys._current_frames().get(h.tid)
        stack = "".join(traceback.format_stack(frame)[-8:]) if frame else "(khong co stack)"
        log.error("STOP_TIMEOUT id=%s tid=%s state=%s alive=True waited=%.1fs stack=\n%s",
                  h.id, h.tid, h.state.value, now - self._stopping_since, stack)
        self._pending_start = False
        self._set_error(h, "stop_timeout", MSG_ZOMBIE, zombie=True)

    def _do_restart(self):
        if not self._spawn("restart", is_restart=True):
            self._set_error(self.worker, "spawn_refused", MSG_STOPPED)

    # ---------------- tao worker ----------------
    def _spawn(self, reason, is_restart=False):
        if self._alive(self.worker) or self.camera.owner is not None:
            log.critical("INVARIANT_VIOLATION spawn refused alive=%s owner=%s",
                         self._alive(self.worker), self.camera.owner)
            return False
        h = WorkerHandle(next(self._ids), self._clock, self.cfg.heartbeat_timeout,
                         self.cfg.open_grace, is_restart)
        h.thread = threading.Thread(target=self._entry, args=(h,), name="worker-%d" % h.id,
                                    daemon=True)
        self.worker = h
        self.phase = self.RUN
        self.intent = None
        log.info("WORKER_LIFECYCLE id=%d tid=- NEW->STARTING reason=%s elapsed=0.00s", h.id, reason)
        h.thread.start()
        return True

    def _entry(self, h):
        h.tid = threading.get_ident()
        try:
            self._worker_main(h)
        except BaseException as exc:
            log.error("WORKER_CRASH id=%s: %r\n%s", h.id, exc, traceback.format_exc())
            h.fail("exception:%s" % type(exc).__name__, restartable=False,
                   msg="SYSTEM ERROR - DETECTION FAILED")
        finally:
            errors = h.run_cleanups()          # landmarker + camera lease dong TRUOC khi bao ket thuc
            h.finish(errors)