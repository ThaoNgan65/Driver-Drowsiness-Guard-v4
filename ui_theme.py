"""
ui_theme.py - Bang mau, font va cac widget dung chung (customtkinter).
"""
import queue
import logging
import customtkinter as ctk

C = {
    "bg": "#0d1117",
    "surface": "#151b23",
    "surface2": "#1c232d",
    "border": "#2a323d",
    "text": "#e8ebf0",
    "muted": "#8b94a3",
    "accent": "#2dd4bf",
    "ok": "#22c55e",
    "ok_bg": "#0f2a1c",
    "warn": "#f59e0b",
    "warn_bg": "#33260a",
    "danger": "#ef4444",
    "danger_bg": "#3a1315",
    "info": "#3b82f6",
    "idle": "#64748b",
}

FAMILY = "Segoe UI"
_fonts = {}


def font(size=13, weight="normal"):
    """CTkFont tao cham (phai co root truoc) va dung lai."""
    key = (size, weight)
    if key not in _fonts:
        _fonts[key] = ctk.CTkFont(family=FAMILY, size=size, weight=weight)
    return _fonts[key]


def mono(size=15, weight="bold"):
    key = ("mono", size, weight)
    if key not in _fonts:
        _fonts[key] = ctk.CTkFont(family="Consolas", size=size, weight=weight)
    return _fonts[key]


def setup():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")


class Card(ctk.CTkFrame):
    """Khung bo goc co tieu de tuy chon."""

    def __init__(self, parent, title=None, **kw):
        kw.setdefault("fg_color", C["surface"])
        kw.setdefault("corner_radius", 14)
        kw.setdefault("border_width", 1)
        kw.setdefault("border_color", C["border"])
        super().__init__(parent, **kw)
        self.title_label = None
        if title:
            self.title_label = ctk.CTkLabel(
                self, text=title.upper(), font=font(11, "bold"),
                text_color=C["muted"], anchor="w")
            self.title_label.pack(fill="x", padx=16, pady=(14, 6))


class Pill(ctk.CTkLabel):
    """Nhan trang thai dang vien thuoc."""

    KINDS = {
        "ok": (C["ok_bg"], C["ok"]),
        "warn": (C["warn_bg"], C["warn"]),
        "danger": (C["danger_bg"], C["danger"]),
        "idle": (C["surface2"], C["muted"]),
        "info": ("#10223d", C["info"]),
    }

    def __init__(self, parent, text="", kind="idle", **kw):
        bg, fg = self.KINDS[kind]
        kw.setdefault("font", font(12, "bold"))
        super().__init__(parent, text=text, fg_color=bg, text_color=fg,
                         corner_radius=999, padx=14, pady=5, **kw)
        self._kind = kind

    def set(self, text, kind):
        if kind != self._kind:
            bg, fg = self.KINDS[kind]
            self.configure(fg_color=bg, text_color=fg)
            self._kind = kind
        self.configure(text=text)


class Tile(ctk.CTkFrame):
    """O so lieu nho: nhan + gia tri."""

    def __init__(self, parent, label, value="—", compact=False):
        super().__init__(parent, fg_color=C["surface2"], corner_radius=10)
        top, bot = (5, 5) if compact else (8, 8)
        ctk.CTkLabel(self, text=label, font=font(10, "bold"),
                     text_color=C["muted"], anchor="w").pack(fill="x", padx=12, pady=(top, 0))
        self.var = ctk.StringVar(value=value)
        self.value_label = ctk.CTkLabel(self, textvariable=self.var,
                                        font=mono(15 if compact else 16),
                                        text_color=C["text"], anchor="w")
        self.value_label.pack(fill="x", padx=12, pady=(0, bot))

    def set(self, text):
        self.var.set(text)


class StatusRow(ctk.CTkFrame):
    """Dong trang thai he thong: cham mau + ten + gia tri."""

    def __init__(self, parent, label):
        super().__init__(parent, fg_color="transparent")
        self.dot = ctk.CTkLabel(self, text="●", font=font(12), text_color=C["idle"], width=16)
        self.dot.pack(side="left")
        ctk.CTkLabel(self, text=label, font=font(12), text_color=C["muted"],
                     anchor="w").pack(side="left", padx=(4, 0))
        self.value = ctk.CTkLabel(self, text="—", font=font(12, "bold"),
                                  text_color=C["text"], anchor="e")
        self.value.pack(side="right")

    def set(self, text, kind="idle"):
        color = {"ok": C["ok"], "warn": C["warn"], "danger": C["danger"],
                 "idle": C["idle"], "info": C["info"]}[kind]
        self.dot.configure(text_color=color)
        self.value.configure(text=text)


def primary_button(parent, text, command, color=None, hover=None, **kw):
    kw.setdefault("height", 40)
    kw.setdefault("corner_radius", 10)
    kw.setdefault("font", font(13, "bold"))
    return ctk.CTkButton(parent, text=text, command=command,
                         fg_color=color or C["accent"], hover_color=hover or "#14b8a6",
                         text_color="#04201c" if color is None else "white", **kw)


def ghost_button(parent, text, command, **kw):
    kw.setdefault("height", 36)
    kw.setdefault("corner_radius", 10)
    kw.setdefault("font", font(12, "bold"))
    return ctk.CTkButton(parent, text=text, command=command, fg_color=C["surface2"],
                         hover_color=C["border"], text_color=C["text"],
                         border_width=1, border_color=C["border"], **kw)


class UiBridge:
    """
    Cau noi an toan tu thread nen sang giao dien.
    Thread nen KHONG duoc goi widget.after()/configure() truc tiep (Tkinter khong an toan da luong);
    thay vao do goi bridge.post(fn, *args) - fn se chay tren main thread.
    """

    def __init__(self, widget, interval_ms=25):
        self.widget = widget
        self.interval = interval_ms
        self.q = queue.Queue()
        self._tick()

    def post(self, fn, *args):
        self.q.put((fn, args))

    def _tick(self):
        try:
            for _ in range(300):
                fn, args = self.q.get_nowait()
                try:
                    fn(*args)
                except Exception as exc:
                    logging.getLogger("drowsiness").warning("UiBridge: %s", exc)
        except queue.Empty:
            pass
        try:
            self.widget.after(self.interval, self._tick)
        except Exception:
            pass          # widget da bi huy