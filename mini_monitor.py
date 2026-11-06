"""
mini_monitor.py - Màn hình giám sát thu nhỏ nổi trên bản đồ (xem camera + trạng thái ngay trong tab Bản đồ).

Chỉ là phần hiển thị; dữ liệu do main.py cập nhật (luôn từ main thread).
"""
import tkinter as tk
import customtkinter as ctk
import ui_theme as T

VIDEO_W, VIDEO_H = 248, 140


class MiniMonitor(ctk.CTkFrame):
    def __init__(self, parent, on_toggle):
        super().__init__(parent, fg_color=T.C["surface"], corner_radius=12,
                         border_width=1, border_color=T.C["border"])
        self.on_toggle = on_toggle
        self.collapsed = False
        self.photo = None

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(head, text="GIÁM SÁT", font=T.font(10, "bold"),
                     text_color=T.C["muted"]).pack(side="left")
        self.collapse_btn = ctk.CTkButton(
            head, text="—", width=26, height=22, corner_radius=6, fg_color=T.C["surface2"],
            hover_color=T.C["border"], text_color=T.C["text"], font=T.font(12, "bold"),
            command=self.toggle_collapse)
        self.collapse_btn.pack(side="right")
        self.pill = T.Pill(head, "ĐÃ DỪNG", "idle")
        self.pill.configure(font=T.font(10, "bold"), padx=10, pady=2)
        self.pill.pack(side="right", padx=(0, 8))

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="x")
        holder = ctk.CTkFrame(self.body, fg_color="#05080c", width=VIDEO_W, height=VIDEO_H, corner_radius=6)
        holder.pack(padx=10, pady=(0, 6))
        holder.pack_propagate(False)
        self.video = tk.Label(holder, bg="#05080c", fg=T.C["muted"], text="Camera đang tắt",
                              font=("Segoe UI", 10), borderwidth=0, highlightthickness=0)
        self.video.pack(fill="both", expand=True)

        row = ctk.CTkFrame(self.body, fg_color="transparent")
        row.pack(fill="x", padx=10)
        ctk.CTkLabel(row, text="Mệt mỏi", font=T.font(11), text_color=T.C["muted"]).pack(side="left")
        self.level = ctk.CTkLabel(row, text="—", font=T.font(11, "bold"), text_color=T.C["muted"])
        self.level.pack(side="right")
        self.bar = ctk.CTkProgressBar(self.body, height=6, corner_radius=3, fg_color=T.C["surface2"],
                                      progress_color=T.C["ok"])
        self.bar.set(0)
        self.bar.pack(fill="x", padx=10, pady=(3, 4))
        self.detail = ctk.CTkLabel(self.body, text="", font=T.font(11), text_color=T.C["warn"],
                                   wraplength=VIDEO_W, justify="left", anchor="w", height=18)
        self.detail.pack(fill="x", padx=10)
        self.btn = T.primary_button(self.body, "▶  Bắt đầu giám sát", self.on_toggle, height=32,
                                    color=T.C["ok"], hover="#16a34a")
        self.btn.configure(text_color="#04150b", font=T.font(12, "bold"))
        self.btn.pack(fill="x", padx=10, pady=(4, 10))

    # ---- vị trí nổi trên bản đồ ----
    def attach(self):
        self.place(relx=0.0, rely=1.0, x=14, y=-14, anchor="sw")
        self.lift()

    def toggle_collapse(self):
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.body.pack_forget()
            self.collapse_btn.configure(text="▢")
        else:
            self.body.pack(fill="x")
            self.collapse_btn.configure(text="—")
        self.lift()

    # ---- cập nhật (main thread) ----
    def set_pill(self, text, kind, detail=""):
        self.pill.set(text, kind)
        self.detail.configure(text=detail[:60])

    def set_fatigue(self, text, color, frac):
        self.level.configure(text=text, text_color=color)
        self.bar.configure(progress_color=color)
        self.bar.set(max(0.0, min(1.0, frac)))

    def set_running(self, running):
        if running:
            self.btn.configure(text="■  Dừng giám sát", fg_color=T.C["danger"], hover_color="#dc2626",
                               text_color="white")
        else:
            self.btn.configure(text="▶  Bắt đầu giám sát", fg_color=T.C["ok"], hover_color="#16a34a",
                               text_color="#04150b")

    def set_frame(self, photo):
        self.photo = photo
        self.video.configure(image=photo, text="")
        self.video.image = photo

    def clear_frame(self):
        self.photo = None
        self.video.configure(image="", text="Camera đang tắt")
        self.video.image = None
