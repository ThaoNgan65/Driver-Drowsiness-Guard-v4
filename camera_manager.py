"""
camera_manager.py - Quan ly QUYEN SO HUU camera (lease) cho Driver Guard.

- Chi module nay goi VideoCapture / read / release (qua factory duoc inject).
- Moi worker xin mot CameraLease (acquire). Lease gan voi (worker_id, gen).
  read()/release() tu lease cu (stale) la no-op co log, khong dong handle cua lease moi.
- Retry/backoff mo camera nam tap trung o day. Vuot gioi han -> CameraUnrecoverable.
- Lock chi bao ve owner/gen, KHONG giu khi open/read => read bi treo khong khoa ai.
- Khong import cv2 o top-level => test duoc bang camera gia.
"""
import logging
import threading
import time

log = logging.getLogger("drowsiness")


class CameraUnrecoverable(Exception):
    """cause: NOT_OPENED | OPEN_EXCEPTION | READ_FAIL_STREAK | OWNER_BUSY | STOP_REQUESTED"""

    def __init__(self, cause, detail=""):
        super().__init__("%s %s" % (cause, detail) if detail else cause)
        self.cause = cause
        self.detail = detail


def default_cv2_factory():
    import cv2                      # import tre: test khong can cv2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraManager:
    def __init__(self, factory=default_cv2_factory, open_retries=3,
                 backoff=(0.5, 1.0, 2.0), fail_reopen=30, open_grace=10.0,
                 read_fail_sleep=0.02, sleep=time.sleep):
        self._factory = factory
        self._retries = max(1, int(open_retries))
        self._backoff = tuple(backoff) or (0.5,)
        self._fail_reopen = max(1, int(fail_reopen))
        self._open_grace = open_grace
        self._read_fail_sleep = read_fail_sleep
        self._sleep = sleep
        self._lock = threading.Lock()
        self._owner = None
        self._gen = 0

    # ---------------- so huu ----------------
    @property
    def owner(self):
        with self._lock:
            return self._owner

    def _is_owner(self, worker_id, gen):
        with self._lock:
            return self._owner == worker_id and self._gen == gen

    def _clear_owner(self, worker_id, gen, reason):
        with self._lock:
            ok = self._owner == worker_id and self._gen == gen
            if ok:
                self._owner = None
        if ok:
            log.info("CAMERA_OWNER release worker=%s gen=%s reason=%s", worker_id, gen, reason)
        else:
            log.warning("STALE_LEASE release ignored worker=%s gen=%s reason=%s",
                        worker_id, gen, reason)
        return ok

    def acquire(self, worker_id, should_stop, beat):
        """Goi tu WORKER thread. should_stop() -> bool; beat(grace=None) cap nhat heartbeat."""
        with self._lock:
            if self._owner is not None:
                log.error("CAMERA_OWNER refuse worker=%s owner=%s", worker_id, self._owner)
                raise CameraUnrecoverable("OWNER_BUSY", "owner=%s" % self._owner)
            self._gen += 1
            gen = self._gen
            self._owner = worker_id
        lease = CameraLease(self, worker_id, gen, should_stop, beat)
        try:
            lease._cap = self._open_with_retry(lease, "acquire")
        except BaseException:
            self._clear_owner(worker_id, gen, "acquire_failed")
            raise
        log.info("CAMERA_OWNER acquire worker=%s gen=%s", worker_id, gen)
        return lease

    # ---------------- mo camera + backoff ----------------
    def _open_with_retry(self, lease, phase):
        cause, detail = "NOT_OPENED", ""
        for attempt in range(1, self._retries + 1):
            if lease._should_stop():
                raise CameraUnrecoverable("STOP_REQUESTED")
            lease._beat(self._open_grace)          # open DSHOW co the mat vai giay
            cap = None
            try:
                cap = self._factory()
                if cap is not None and cap.isOpened():
                    return cap
                cause, detail = "NOT_OPENED", "isOpened=False"
            except Exception as exc:
                cause, detail = "OPEN_EXCEPTION", repr(exc)
            if cap is not None:
                try:
                    cap.release()
                except Exception as exc:
                    log.warning("CAMERA_RELEASE_ERROR worker=%s: %s", lease.worker_id, exc)
            if attempt < self._retries:
                wait_s = self._backoff[min(attempt - 1, len(self._backoff) - 1)]
                log.warning("CAMERA_RETRY worker=%s gen=%s phase=%s attempt=%d/%d cause=%s backoff=%.2fs",
                            lease.worker_id, lease.gen, phase, attempt, self._retries, cause, wait_s)
                if not self._wait(wait_s, lease):
                    raise CameraUnrecoverable("STOP_REQUESTED")
            else:
                log.error("CAMERA_RETRY worker=%s gen=%s phase=%s attempt=%d/%d cause=%s giveup",
                          lease.worker_id, lease.gen, phase, attempt, self._retries, cause)
        raise CameraUnrecoverable(cause, detail)

    def _wait(self, seconds, lease):
        """Cho theo lat 0.1s: van beat heartbeat va kiem tra stop."""
        end = time.monotonic() + seconds
        while True:
            if lease._should_stop():
                return False
            left = end - time.monotonic()
            if left <= 0:
                return True
            lease._beat(self._open_grace)
            self._sleep(min(0.1, left))


class CameraLease:
    """Token so huu camera. Chi worker_id nay duoc dung; chi thread cua worker goi read/release."""

    def __init__(self, mgr, worker_id, gen, should_stop, beat):
        self._mgr = mgr
        self.worker_id = worker_id
        self.gen = gen
        self._should_stop = should_stop
        self._beat = beat
        self._cap = None
        self._released = False
        self._fail_reads = 0
        self._episodes = 0            # so dot reconnect lien tiep chua co frame nao
        self._stale_logged = False
        self._last_reopen_try = 0.0

    @property
    def active(self):
        return (not self._released) and self._mgr._is_owner(self.worker_id, self.gen)

    def read(self):
        """-> (ok, frame). Mat camera: tu reconnect (backoff). Vuot gioi han -> CameraUnrecoverable."""
        if not self.active:
            if not self._stale_logged:
                self._stale_logged = True
                log.warning("STALE_LEASE read ignored worker=%s gen=%s", self.worker_id, self.gen)
            self._mgr._sleep(self._mgr._read_fail_sleep)
            return False, None
        if self._cap is None:
            # Lan mo lai truoc do that bai (camera dang bi app khac giu): thu lai moi 2s,
            # van beat de watchdog khong restart worker vi "khong co frame".
            self._beat()
            if time.monotonic() - self._last_reopen_try >= 2.0:
                self.reopen("no_handle")
            else:
                self._mgr._sleep(0.05)
            return False, None
        try:
            ret, frame = self._cap.read()
        except Exception as exc:
            log.warning("CAMERA_READ_ERROR worker=%s: %s", self.worker_id, exc)
            ret, frame = False, None
        if ret:
            self._fail_reads = 0
            self._episodes = 0
            self._beat()
            return True, frame
        self._fail_reads += 1
        if self._fail_reads >= self._mgr._fail_reopen:
            log.warning("CAMERA_RETRY worker=%s gen=%s cause=READ_FAIL_STREAK reads=%d",
                        self.worker_id, self.gen, self._fail_reads)
            self._reconnect()
        else:
            self._mgr._sleep(self._mgr._read_fail_sleep)
        return False, None

    def _reconnect(self):
        self._episodes += 1
        if self._episodes > self._mgr._retries:
            raise CameraUnrecoverable(
                "READ_FAIL_STREAK", "%d lan reconnect khong co frame" % (self._episodes - 1))
        old, self._cap = self._cap, None
        if old is not None:
            try:
                old.release()
            except Exception as exc:
                log.warning("CAMERA_RELEASE_ERROR worker=%s: %s", self.worker_id, exc)
        self._cap = self._mgr._open_with_retry(self, "reconnect")
        self._fail_reads = 0

    def reopen(self, reason="dead_frames"):
        """Dong handle hien tai va mo lai. KHONG nem loi khi camera dang bi app khac giu:
        tra False va de _cap=None, read() se tu thu lai theo chu ky. Chi nem
        CameraUnrecoverable("STOP_REQUESTED") khi worker bi yeu cau dung."""
        if not self.active:
            return False
        log.warning("CAMERA_REOPEN worker=%s gen=%s reason=%s", self.worker_id, self.gen, reason)
        old, self._cap = self._cap, None
        if old is not None:
            try:
                old.release()
            except Exception as exc:
                log.warning("CAMERA_RELEASE_ERROR worker=%s: %s", self.worker_id, exc)
        self._last_reopen_try = time.monotonic()
        if not self._mgr._wait(0.3, self):          # DSHOW can chut thoi gian nha handle
            raise CameraUnrecoverable("STOP_REQUESTED")
        try:
            self._cap = self._mgr._open_with_retry(self, "reopen")
        except CameraUnrecoverable as exc:
            if exc.cause == "STOP_REQUESTED":
                raise
            log.warning("CAMERA_REOPEN_FAILED worker=%s cause=%s - se thu lai sau", self.worker_id, exc.cause)
            self._cap = None
            return False
        self._fail_reads = 0
        self._episodes = 0
        log.info("CAMERA_REOPEN_OK worker=%s gen=%s", self.worker_id, self.gen)
        return True

    def release(self, reason="release"):
        """Idempotent. Dong handle CUA LEASE NAY roi moi tra owner (owner=None => handle da dong)."""
        if self._released:
            log.warning("STALE_LEASE release ignored worker=%s gen=%s (da release)",
                        self.worker_id, self.gen)
            return
        self._released = True
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                log.warning("CAMERA_RELEASE_ERROR worker=%s: %s", self.worker_id, exc)
        self._mgr._clear_owner(self.worker_id, self.gen, reason)