# route_map.py
# Ban do va lo trinh hien thi truc tiep trong Tkinter
# Khong tu dong mo trinh duyet

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import requests
import tkintermapview

import os
import re
import json
import math
import time
import serial
import serial.tools.list_ports


import asyncio

LOCATE_STATUS = "ok"      # ly do dinh vi that bai gan nhat: ok | denied | no_fix | unsupported
LOCATE_TIMEOUT_S = 10


def locate_by_windows():
    """Lay vi tri tu Windows Location, co the su dung Wi-Fi.
    Tra ve None neu that bai; ly do nam o LOCATE_STATUS."""
    global LOCATE_STATUS
    try:
        from winsdk.windows.devices.geolocation import (
            Geolocator,
            GeolocationAccessStatus,
        )

        async def get_position():
            permission = await Geolocator.request_access_async()

            if permission != GeolocationAccessStatus.ALLOWED:
                return None, "denied"

            locator = Geolocator()
            position = await asyncio.wait_for(
                locator.get_geoposition_async(), LOCATE_TIMEOUT_S
            )

            coords = position.coordinate.point.position

            return {
                "lat": coords.latitude,
                "lon": coords.longitude,
                "source": "Windows Location",
            }, "ok"

        result, LOCATE_STATUS = asyncio.run(get_position())
        return result

    except ImportError:
        LOCATE_STATUS = "unsupported"
    except asyncio.TimeoutError:
        LOCATE_STATUS = "no_fix"
    except Exception as e:
        LOCATE_STATUS = "denied"
        print(f"Windows Location loi: {e}")
    return None


def parse_nmea_coordinate(value, direction):
    """Chuyen toa do NMEA sang so thap phan."""
    if not value:
        return None

    try:
        raw = float(value)
        degrees = int(raw // 100)
        minutes = raw - degrees * 100
        coordinate = degrees + minutes / 60

        if direction in ("S", "W"):
            coordinate = -coordinate

        return coordinate
    except (ValueError, TypeError):
        return None


def read_gps_from_port(port, baudrate=9600, timeout=1):
    """Doc NMEA tu mot cong serial; tra ve toa do neu co fix."""
    try:
        with serial.Serial(
            port,
            baudrate=baudrate,
            timeout=timeout
        ) as gps:
            deadline = time.time() + 5

            while time.time() < deadline:
                line = gps.readline().decode(
                    "ascii", errors="ignore"
                ).strip()

                if not line.startswith("$"):
                    continue

                # GNRMC / GPRMC: chi chap nhan status A.
                if "RMC," in line:
                    fields = line.split(",")
                    if len(fields) > 6 and fields[2] == "A":
                        lat = parse_nmea_coordinate(
                            fields[3], fields[4]
                        )
                        lon = parse_nmea_coordinate(
                            fields[5], fields[6]
                        )

                        if lat is not None and lon is not None:
                            return {
                                "lat": lat,
                                "lon": lon,
                                "name": "GPS USB",
                                "source": "GPS",
                            }

                # GGA: quality > 0 nghia la co GPS fix.
                if "GGA," in line:
                    fields = line.split(",")
                    if len(fields) > 6 and fields[6].isdigit():
                        if int(fields[6]) > 0:
                            lat = parse_nmea_coordinate(
                                fields[2], fields[3]
                            )
                            lon = parse_nmea_coordinate(
                                fields[4], fields[5]
                            )

                            if lat is not None and lon is not None:
                                return {
                                    "lat": lat,
                                    "lon": lon,
                                    "name": "GPS USB",
                                    "source": "GPS",
                                }

    except (serial.SerialException, OSError):
        pass

    return None


def detect_gps():
    """Tu dong thu doc cac cong serial thuong dung."""
    ports = list(serial.tools.list_ports.comports())

    # Thu cac cong co mo ta giong GPS truoc.
    ports.sort(
        key=lambda p: 0 if "gps" in (
            (p.description or "").lower()
        ) else 1
    )

    for port in ports:
        result = read_gps_from_port(port.device)
        if result:
            return result

    return None



def locate_by_ip():
    """
    Dinh vi gan dung qua IP.
    Thu nhieu dich vu lan luot neu mot dich vu loi.
    Luu y: vi tri IP khong chinh xac nhu GPS.
    """
    providers = [
        {
            "url": "https://ipwho.is/",
            "lat_key": "latitude",
            "lon_key": "longitude",
            "success_key": "success",
        },
        {
            "url": "https://ipapi.co/json/",
            "lat_key": "latitude",
            "lon_key": "longitude",
            "success_key": None,
        },
    ]

    errors = []

    for provider in providers:
        try:
            response = requests.get(
                provider["url"],
                headers={"User-Agent": "RoutePlanner/1.0"},
                timeout=8,
            )

            if response.status_code == 429:
                errors.append(
                    provider["url"] + ": qua nhieu yeu cau (429)"
                )
                continue

            response.raise_for_status()
            data = response.json()

            success_key = provider["success_key"]
            if (
                success_key
                and data.get(success_key) is not True
            ):
                errors.append(
                    provider["url"] + ": dich vu khong tra ve vi tri"
                )
                continue

            lat = data.get(provider["lat_key"])
            lon = data.get(provider["lon_key"])

            if lat is None or lon is None:
                errors.append(
                    provider["url"] + ": khong co toa do"
                )
                continue

            lat = float(lat)
            lon = float(lon)

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                errors.append(
                    provider["url"] + ": toa do khong hop le"
                )
                continue

            return {
                "lat": lat,
                "lon": lon,
                "source": "IP (ước lượng)",
            }

        except Exception as exc:
            errors.append(
                f"{provider['url']}: {exc}"
            )

    raise RuntimeError(
        "Không thể định vị qua IP từ các dịch vụ dự phòng.\n"
        + "\n".join(errors)
        + "\nHãy thử lại sau hoặc nhập điểm đi thủ công."
    )



def auto_locate(allow_ip=True):
    """
    Uu tien Windows Location (Wi-Fi),
    sau do GPS USB, cuoi cung la IP (bo qua neu allow_ip=False).
    """

    # 1. Dinh vi bang Windows
    result = locate_by_windows()
    if result:
        print("Da lay vi tri tu Windows Location")
        return result

    # 2. Thu GPS USB
    result = detect_gps()
    if result:
        print("Da lay vi tri tu GPS")
        return result

    # 3. Phuong an du phong: IP (khong chinh xac, hay sai thanh pho)
    if not allow_ip:
        return None
    print("Khong co vi tri Windows/GPS, thu dinh vi IP")
    return locate_by_ip()


LOCATION_HELP = {
    "denied": (
        "Mình chưa lấy được vị trí vì định vị trên máy đang tắt hoặc chưa cho phép ứng dụng. "
        "Bạn bật như sau nhé:\n"
        "1. Mở Cài đặt Windows → Quyền riêng tư & bảo mật → Vị trí "
        "(Windows 10: Cài đặt → Quyền riêng tư → Vị trí).\n"
        "2. Bật «Dịch vụ vị trí» (Location services).\n"
        "3. Bật «Cho phép ứng dụng truy cập vị trí của bạn», và bật cả «Cho phép ứng dụng trên máy tính "
        "truy cập vị trí của bạn» (Let desktop apps access your location), vì đây là ứng dụng máy tính.\n"
        "4. Hỏi lại mình về thời tiết.\n"
        "Hoặc bạn cứ nói tên tỉnh/thành, ví dụ «thời tiết Đà Nẵng»."
    ),
    "no_fix": (
        "Mình chưa nhận được tín hiệu định vị. Nếu dùng GPS USB, hãy cắm thiết bị, đưa ra nơi thoáng "
        "và chờ vài phút để bắt tín hiệu; nếu dùng định vị của Windows, hãy kiểm tra mạng đang bật "
        "và «Dịch vụ vị trí» đã mở. Hoặc bạn nói tên tỉnh/thành, ví dụ «thời tiết Đà Nẵng»."
    ),
    "unsupported": (
        "Máy này chưa có nguồn định vị khả dụng (Windows Location hoặc GPS USB). "
        "Bạn có thể nói tên tỉnh/thành để mình xem thời tiết, ví dụ «thời tiết Đà Nẵng»."
    ),
}

_precise = {"t": 0.0, "loc": None}
_precise_lock = threading.Lock()


def locate_precise(max_age=120):
    """
    Vi tri cho tro ly (thoi tiet...): Windows Location -> GPS USB, KHONG dung IP.
    Tra ve (vi_tri | None, huong_dan). huong_dan chi co noi dung khi vi_tri la None.
    Chi luu dem khi thanh cong, nen vua bat dinh vi xong la hoi lai duoc ngay.
    """
    with _precise_lock:
        if _precise["loc"] and time.time() - _precise["t"] < max_age:
            return _precise["loc"], ""
        loc = auto_locate(allow_ip=False)
        if loc:
            _precise.update(t=time.time(), loc=loc)
            return loc, ""
        return None, LOCATION_HELP.get(LOCATE_STATUS, LOCATION_HELP["denied"])


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

HEADERS = {
    "User-Agent": "DriverSafetyRoutePlanner/1.0"
}

CATEGORIES = {
    "Tất cả địa điểm": "all",
    "Trạm xăng": "fuel",
    "Ăn uống / cà phê": "food",
    "Bệnh viện / y tế": "medical",
    "Điểm nghỉ / bãi đỗ": "rest",
}

CATEGORY_LABELS = {
    "fuel": "Trạm xăng",
    "food": "Ăn uống / cà phê",
    "medical": "Bệnh viện / y tế",
    "rest": "Điểm nghỉ / bãi đỗ",
}


def geocode(query):
    """Chuyen ten dia diem hoac toa do thanh vi tri."""
    query = query.strip()

    if not query:
        raise ValueError("Vui lòng nhập địa điểm.")

    # Ho tro nhap toa do: 10.0452, 105.7469
    try:
        parts = [float(x.strip()) for x in query.split(",")]
        if (
            len(parts) == 2
            and -90 <= parts[0] <= 90
            and -180 <= parts[1] <= 180
        ):
            return {
                "lat": parts[0],
                "lon": parts[1],
                "name": query,
            }
    except ValueError:
        pass

    return _geocode_remote(query)


def _http_get_json(url, params, timeout=(6, 20), tries=3):
    """GET co thu lai (ngat ket noi 10054, 5xx, 429) voi backoff ngan."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}")
            else:
                r.raise_for_status()
                return r.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
        time.sleep(0.6 * (i + 1))
    raise last


def _geocode_remote(query):
    """Geoapify (neu co key) -> Nominatim -> Photon. Dich vu nao loi thi thu dich vu ke tiep."""
    import settings
    errors = []
    key = settings.get("GEOAPIFY_API_KEY", "")

    if key:
        try:
            d = _http_get_json("https://api.geoapify.com/v1/geocode/search",
                               {"text": query, "limit": 1, "lang": "vi", "apiKey": key},
                               tries=2)
            feats = d.get("features") or []
            if feats:
                lon, lat = feats[0]["geometry"]["coordinates"][:2]
                return {"lat": float(lat), "lon": float(lon),
                        "name": feats[0]["properties"].get("formatted", query)}
            errors.append("Geoapify: khong co ket qua")
        except Exception as exc:
            errors.append(f"Geoapify: {exc}")

    try:
        res = _http_get_json(NOMINATIM_URL,
                             {"q": query, "format": "jsonv2", "limit": 1}, tries=2)
        if res:
            return {"lat": float(res[0]["lat"]), "lon": float(res[0]["lon"]),
                    "name": res[0].get("display_name", query)}
        errors.append("Nominatim: khong co ket qua")
    except Exception as exc:
        errors.append(f"Nominatim: {exc}")

    try:
        d = _http_get_json("https://photon.komoot.io/api/",
                           {"q": query, "limit": 1}, tries=2)
        feats = d.get("features") or []
        if feats:
            lon, lat = feats[0]["geometry"]["coordinates"][:2]
            pr = feats[0].get("properties", {})
            name = ", ".join(x for x in (pr.get("name"), pr.get("state"), pr.get("country")) if x)
            return {"lat": float(lat), "lon": float(lon), "name": name or query}
        errors.append("Photon: khong co ket qua")
    except Exception as exc:
        errors.append(f"Photon: {exc}")

    if all("khong co ket qua" in e for e in errors):
        raise ValueError(f"Không tìm thấy địa điểm: {query}")
    raise ConnectionError("Không kết nối được dịch vụ tìm địa điểm (" + "; ".join(errors) + ")")


def calculate_route(origin, destination):
    """Tinh tuyen duong: Geoapify (neu co key) -> OSRM, co thu lai."""
    import settings
    errors = []
    key = settings.get("GEOAPIFY_API_KEY", "")

    if key:
        try:
            d = _http_get_json(
                "https://api.geoapify.com/v1/routing",
                {"waypoints": f"{origin['lat']},{origin['lon']}|{destination['lat']},{destination['lon']}",
                 "mode": "drive", "type": "short", "apiKey": key}, timeout=(6, 30), tries=2)
            f = d["features"][0]
            pts = []
            for line in f["geometry"]["coordinates"]:
                if line and isinstance(line[0], (int, float)):
                    line = [line]
                pts.extend((lat, lon) for lon, lat in line)
            pr = f["properties"]
            if pts:
                return {"points": pts, "distance": pr["distance"], "duration": pr["time"]}
        except Exception as exc:
            errors.append(f"Geoapify: {exc}")

    coords = (f"{origin['lon']},{origin['lat']};"
              f"{destination['lon']},{destination['lat']}")
    try:
        data = _http_get_json(f"{OSRM_URL}/{coords}",
                              {"overview": "full", "geometries": "geojson", "steps": "false", "alternatives": "true"},
                              timeout=(6, 35), tries=3)
        if data.get("code") == "Ok" and data.get("routes"):
            route = min(data["routes"], key=lambda r: r["distance"])
            return {"points": [(lat, lon) for lon, lat in route["geometry"]["coordinates"]],
                    "distance": route["distance"], "duration": route["duration"]}
        errors.append("OSRM: khong co tuyen")
    except Exception as exc:
        errors.append(f"OSRM: {exc}")

    raise ConnectionError("Không tính được tuyến đường (" + "; ".join(errors) + ")")


# Hinh hoc: geo_utils. Tim dia diem: poi_provider (Geoapify / Google / Overpass du phong).
from geo_utils import (  # noqa: E402,F401
    haversine_m, sample_route, places_ahead_on_route,
)
import poi_provider  # noqa: E402


def find_places(route_points, category, radius=1500):
    """Dia diem doc tuyen cho ban do. Tra [] neu loi (van ve duoc tuyen)."""
    return poi_provider.find_along_route(route_points, category, radius)


def find_places_near(lat, lon, category="all", radius=3000):
    """Dia diem quanh mot vi tri, co 'dist_m'. Nem RuntimeError neu khong co nguon nao."""
    return poi_provider.find_near(lat, lon, category, radius)


def format_duration(seconds):
    minutes = round(seconds / 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours} giờ {minutes} phút"

    return f"{minutes} phút"



class RoutePlanner:
    """
    Ban do + lo trinh. container=None: mo cua so rieng (Toplevel).
    container=<frame>: nhung thang vao trang cua ung dung chinh.
    """

    def __init__(self, parent, container=None):
        global _planner
        import customtkinter as ctk
        import ui_theme as T

        self.parent = parent
        self.embedded = container is not None
        if self.embedded:
            self.window = container
        else:
            self.window = ctk.CTkToplevel(parent)
            self.window.title("Bản đồ và lộ trình")
            self.window.geometry("1100x740")
            self.window.minsize(860, 600)
            self.window.configure(fg_color=T.C["bg"])
        _planner = self
        self.ui = T.UiBridge(self.window)

        self.busy = False
        self.places = []
        self.origin = None
        self.destination = None
        self.route = None

        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(2, weight=1)

        # ===== FORM =====
        form = T.Card(self.window)
        form.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
        form.grid_columnconfigure(1, weight=1)
        self.plan = None

        def label(text, row):
            ctk.CTkLabel(form, text=text, font=T.font(12, "bold"),
                         text_color=T.C["muted"], width=128, anchor="w"
                         ).grid(row=row, column=0, padx=(16, 6), pady=8, sticky="w")

        entry_kw = dict(height=38, corner_radius=10, border_width=1,
                        fg_color=T.C["surface2"], border_color=T.C["border"],
                        text_color=T.C["text"], font=T.font(13))
        label("Điểm đi", 0)
        self.origin_entry = ctk.CTkEntry(
            form, placeholder_text="Địa chỉ hoặc toạ độ (vd: 10.0452, 105.7469)", **entry_kw)
        self.origin_entry.grid(row=0, column=1, padx=(0, 8), pady=(14, 4), sticky="ew")
        self.locate_button = T.ghost_button(form, "Định vị tôi", self.auto_locate_current, width=120)
        self.locate_button.grid(row=0, column=2, padx=(0, 16), pady=(14, 4))

        label("Điểm đến", 1)
        self.destination_entry = ctk.CTkEntry(form, placeholder_text="Nhập điểm đến", **entry_kw)
        self.destination_entry.grid(row=1, column=1, padx=(0, 8), pady=4, sticky="ew")
        self.search_button = T.primary_button(
            form, "Tìm lộ trình", self.search, width=120)
        self.search_button.grid(row=1, column=2, padx=(0, 16), pady=4)

        label("Gợi ý dọc đường", 2)
        self.category_var = ctk.StringVar(value="Tất cả địa điểm")
        ctk.CTkOptionMenu(
            form, variable=self.category_var, values=list(CATEGORIES.keys()),
            height=36, corner_radius=10, fg_color=T.C["surface2"],
            button_color=T.C["border"], button_hover_color=T.C["info"],
            text_color=T.C["text"], font=T.font(12), dropdown_font=T.font(12),
            anchor="w", width=240,
        ).grid(row=2, column=1, padx=(0, 8), pady=(4, 14), sticky="w")
        T.ghost_button(form, "Xóa gợi ý", self.clear_places, width=120
                       ).grid(row=2, column=2, padx=(0, 16), pady=(4, 14))

        # --- Ke hoach di chuyen an toan ---
        label("Kế hoạch an toàn", 3)
        plan_row = ctk.CTkFrame(form, fg_color="transparent")
        plan_row.grid(row=3, column=1, columnspan=2, sticky="w", pady=(0, 14))
        self.plan_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(plan_row, text="Lập kế hoạch nghỉ", variable=self.plan_var,
                      font=T.font(12), text_color=T.C["text"],
                      progress_color=T.C["accent"]).pack(side="left", padx=(0, 18))
        ctk.CTkLabel(plan_row, text="Giờ khởi hành", font=T.font(12),
                     text_color=T.C["muted"]).pack(side="left", padx=(0, 6))
        self.depart_entry = ctk.CTkEntry(plan_row, width=84, height=34, corner_radius=8,
                                         border_width=1, fg_color=T.C["surface2"],
                                         border_color=T.C["border"], text_color=T.C["text"],
                                         font=T.font(12), placeholder_text="HH:MM")
        self.depart_entry.pack(side="left", padx=(0, 18))
        ctk.CTkLabel(plan_row, text="Đã lái trước đó (phút)", font=T.font(12),
                     text_color=T.C["muted"]).pack(side="left", padx=(0, 6))
        self.driven_entry = ctk.CTkEntry(plan_row, width=64, height=34, corner_radius=8,
                                         border_width=1, fg_color=T.C["surface2"],
                                         border_color=T.C["border"], text_color=T.C["text"],
                                         font=T.font(12), placeholder_text="0")
        self.driven_entry.pack(side="left")

        # ===== TRANG THAI =====
        self.status = ctk.StringVar(value="Sẵn sàng.")
        ctk.CTkLabel(self.window, textvariable=self.status, font=T.font(12),
                     text_color=T.C["muted"], anchor="w", justify="left", wraplength=900
                     ).grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 8))

        # ===== BAN DO + DANH SACH =====
        content = ctk.CTkFrame(self.window, fg_color="transparent")
        content.grid(row=2, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)

        map_card = T.Card(content, corner_radius=14)
        map_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.map_card = map_card
        self.map_widget = tkintermapview.TkinterMapView(map_card, corner_radius=12)
        self.map_widget.set_tile_server(
            "https://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png", max_zoom=19)
        self.map_widget.pack(fill="both", expand=True, padx=6, pady=6)
        self.map_widget.set_position(10.0452, 105.7469)
        self.map_widget.set_zoom(7)

        side = ctk.CTkFrame(content, fg_color="transparent", width=340)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.pack_propagate(False)

        tabs = ctk.CTkTabview(
            side, height=330, corner_radius=14, fg_color=T.C["surface"],
            border_width=1, border_color=T.C["border"],
            segmented_button_fg_color=T.C["surface2"],
            segmented_button_selected_color=T.C["accent"],
            segmented_button_selected_hover_color="#14b8a6",
            segmented_button_unselected_color=T.C["surface2"],
            segmented_button_unselected_hover_color=T.C["border"],
            text_color=T.C["text"])
        tabs.pack(fill="x", pady=(0, 10))
        tabs.pack_propagate(False)
        tab_plan = tabs.add("Kế hoạch an toàn")
        tab_agent = tabs.add("Agent AI")
        self._tabs = tabs
        self.trip_hooks = {}
        trip_row = ctk.CTkFrame(tab_plan, fg_color="transparent")
        trip_row.pack(side="bottom", fill="x", pady=(6, 0))
        self.trip_btn = T.primary_button(trip_row, "Bắt đầu hành trình", self.toggle_trip, height=34)
        self.trip_btn.pack(fill="x")
        self.trip_btn.configure(state="disabled")
        self.plan_box = ctk.CTkTextbox(
            tab_plan, fg_color=T.C["surface"], text_color=T.C["text"],
            font=T.font(12), wrap="word", border_width=0, activate_scrollbars=True)
        self.plan_box.pack(fill="both", expand=True)
        self._plan_set("Bật «Lập kế hoạch nghỉ», nhập điểm đến rồi bấm Tìm lộ trình.")

        # --- Agent AI (Gemini function calling) ---
        self.agent = None
        self.agent_busy = False
        self.position_getter = lambda: None
        self.rag_getter = lambda: None
        self.agent_log = ctk.CTkTextbox(
            tab_agent, fg_color=T.C["surface"], text_color=T.C["text"],
            font=T.font(12), wrap="word", border_width=0, activate_scrollbars=True)
        self.agent_log.pack(fill="both", expand=True)
        self.agent_log._textbox.tag_configure("me", foreground=T.C["info"])
        self.agent_log._textbox.tag_configure("tool", foreground=T.C["muted"])
        self.agent_log._textbox.tag_configure("bot", foreground=T.C["text"])
        self.agent_status = ctk.StringVar(value="")
        ctk.CTkLabel(tab_agent, textvariable=self.agent_status, font=T.font(11),
                     text_color=T.C["muted"], anchor="w").pack(fill="x", pady=(4, 0))
        arow = ctk.CTkFrame(tab_agent, fg_color="transparent")
        arow.pack(fill="x", pady=(2, 0))
        self.agent_entry = ctk.CTkEntry(
            arow, height=34, corner_radius=8, border_width=1, fg_color=T.C["surface2"],
            border_color=T.C["border"], text_color=T.C["text"], font=T.font(12),
            placeholder_text="VD: Mình đi Hà Nội, 7h xuất phát")
        self.agent_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.agent_entry.bind("<Return>", lambda _e: self.agent_send())
        self.agent_send_btn = T.primary_button(arow, "Gửi", self.agent_send, width=64, height=34)
        self.agent_send_btn.pack(side="left")
        self._agent_write("Agent AI: nói chuyện bình thường, ví dụ «lên kế hoạch đi Vĩnh Long, mình đã lái 2 tiếng rồi» "
                          "hoặc «buồn ngủ thì nên làm gì?». Agent tự gọi công cụ (tìm đường, tìm điểm dừng, "
                          "tra tài liệu) và chỉ dùng số liệu từ công cụ.\n", "tool")

        list_card = T.Card(side, title="Địa điểm gợi ý")
        list_card.pack(fill="both", expand=True)
        list_card.pack_propagate(False)
        lb_frame = ctk.CTkFrame(list_card, fg_color="transparent")
        lb_frame.pack(fill="both", expand=True, padx=10, pady=(0, 12))
        scrollbar = ctk.CTkScrollbar(lb_frame, orientation="vertical")
        scrollbar.pack(side="right", fill="y")
        self.place_list = tk.Listbox(
            lb_frame, yscrollcommand=scrollbar.set, font=("Segoe UI", 10),
            bg=T.C["surface"], fg=T.C["text"], selectbackground=T.C["info"],
            selectforeground="white", highlightthickness=0, relief="flat",
            activestyle="none", borderwidth=0)
        self.place_list.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self.place_list.yview)
        self.place_list.bind("<<ListboxSelect>>", self.focus_selected_place)

        self.summary = ctk.StringVar(value="Chưa có lộ trình.")
        ctk.CTkLabel(self.window, textvariable=self.summary, font=T.font(12),
                     text_color=T.C["text"], anchor="w", justify="left", wraplength=900
                     ).grid(row=3, column=0, sticky="ew", padx=4, pady=(8, 0))

    # ==================================================
    # TU DINH VI: Windows -> GPS USB -> IP
    # ==================================================
    def auto_locate_current(self):
        if self.busy:
            return

        self.busy = True
        self.locate_button.configure(state="disabled")
        self.search_button.configure(state="disabled")

        self.status.set(
            "Đang định vị (Windows Location → GPS USB → IP)..."
        )

        def worker():
            try:
                # Ham auto_locate() duoc khai bao ben ngoai class.
                location = auto_locate()

                self.ui.post(lambda loc=location: self.on_location_found(loc))

            except Exception as exc:
                message = str(exc)

                self.ui.post(lambda msg=message: self.on_location_error(msg))

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    def on_location_found(self, location):
        if not self.window.winfo_exists():
            return

        self.busy = False
        self.locate_button.configure(state="normal")
        self.search_button.configure(state="normal")

        lat = float(location["lat"])
        lon = float(location["lon"])
        source = location.get("source", "Không rõ")

        # Dien toa do vao o diem di.
        self.origin_entry.delete(0, tk.END)
        self.origin_entry.insert(
            0, f"{lat:.6f}, {lon:.6f}"
        )

        # Luu vi tri vua lay duoc.
        self.origin = {
            "lat": lat,
            "lon": lon,
            "name": "Vị trí hiện tại",
        }

        # Xoa hien thi cu, sau do danh dau vi tri moi.
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()

        self.map_widget.set_position(lat, lon)
        self.map_widget.set_zoom(15)

        self.map_widget.set_marker(
            lat,
            lon,
            text=f"Vị trí hiện tại ({source})",
        )

        self.status.set(
            f"Đã định vị bằng {source}: "
            f"{lat:.6f}, {lon:.6f}. "
            "Nhập điểm đến rồi bấm Tìm lộ trình."
        )

        self.summary.set(
            "Đã xác định điểm đi. Chưa tính lộ trình."
        )

    def on_location_error(self, message):
        if not self.window.winfo_exists():
            return

        self.busy = False
        self.locate_button.configure(state="normal")
        self.search_button.configure(state="normal")

        self.status.set("Không thể tự định vị.")

        messagebox.showwarning(
            "Không định vị được",
            "Không tìm được GPS hợp lệ và định vị IP cũng thất bại.\n\n"
            f"Chi tiết: {message}\n\n"
            "Bạn có thể nhập điểm đi thủ công.",
            parent=self.window.winfo_toplevel(),
        )

    # ==================================================
    # CHI DUONG TU BANG CANH BAO (dien san diem di/den)
    # ==================================================
    def route_to(self, dest_lat, dest_lon, dest_name="", origin=None):
        """Dien san diem di/den roi tim lo trinh. origin = {'lat','lon'}."""
        try:
            top = self.window.winfo_toplevel()
            top.deiconify()
            top.lift()
        except Exception:
            pass

        if origin:
            self.origin_entry.delete(0, tk.END)
            self.origin_entry.insert(
                0, f"{float(origin['lat']):.6f}, {float(origin['lon']):.6f}"
            )

        self.destination_entry.delete(0, tk.END)
        self.destination_entry.insert(
            0, f"{float(dest_lat):.6f}, {float(dest_lon):.6f}"
        )
        self.category_var.set("Tất cả địa điểm")
        self.status.set(
            f"Đang chỉ đường đến: {dest_name}" if dest_name
            else "Đang chỉ đường..."
        )
        self.search()

    # ==================================================
    # TIM LO TRINH VA DIA DIEM
    # ==================================================
    def search(self):
        if self.busy:
            return

        origin_text = self.origin_entry.get().strip()
        destination_text = (
            self.destination_entry.get().strip()
        )

        if not origin_text or not destination_text:
            messagebox.showwarning(
                "Thiếu thông tin",
                "Hãy nhập điểm đi và điểm đến.",
                parent=self.window.winfo_toplevel(),
            )
            return

        category_name = self.category_var.get()

        if category_name not in CATEGORIES:
            messagebox.showwarning(
                "Danh mục không hợp lệ",
                "Vui lòng chọn danh mục địa điểm.",
                parent=self.window.winfo_toplevel(),
            )
            return

        category = CATEGORIES[category_name]

        want_plan = bool(self.plan_var.get())
        try:
            driven_min = max(0.0, float(self.driven_entry.get().strip() or 0))
        except ValueError:
            messagebox.showwarning("Dữ liệu chưa hợp lệ",
                                   "«Đã lái trước đó» phải là số phút (vd: 90).",
                                   parent=self.window.winfo_toplevel())
            return
        depart = self._parse_depart(self.depart_entry.get())
        if depart is None:
            messagebox.showwarning("Dữ liệu chưa hợp lệ",
                                   "Giờ khởi hành nhập dạng HH:MM (vd: 07:30) hoặc để trống = bây giờ.",
                                   parent=self.window.winfo_toplevel())
            return

        self.busy = True
        self.search_button.configure(state="disabled")
        self.locate_button.configure(state="disabled")

        self.status.set(
            "Đang tìm vị trí và tính lộ trình..."
        )

        def worker():
            try:
                origin = geocode(origin_text)
                destination = geocode(destination_text)

                route = calculate_route(
                    origin,
                    destination,
                )

                # Loi tim POI khong lam mat tuyen duong.
                try:
                    places = find_places(
                        route["points"],
                        category,
                    )
                    places_warning = None

                except Exception as exc:
                    places = []
                    places_warning = str(exc)

                plan, plan_text, plan_src, plan_error = None, "", "", ""
                if want_plan:
                    try:
                        import trip_planner
                        plan = trip_planner.build_plan(
                            route,
                            lambda la, lo: find_places_near(
                                la, lo, "all", trip_planner.SEARCH_RADIUS_M),
                            depart=depart, already_driven_min=driven_min)
                        try:
                            from fatigue_advisor import GeminiPhraser
                            plan_text, plan_src = trip_planner.phrase_summary(plan, GeminiPhraser())
                        except Exception:
                            plan_text, plan_src = trip_planner.template_summary(plan), "template"
                    except Exception as exc:
                        plan_error = f"{type(exc).__name__}: {exc}"

                result = {
                    "origin": origin,
                    "destination": destination,
                    "route": route,
                    "places": places,
                    "places_warning": places_warning,
                    "plan": plan, "plan_text": plan_text,
                    "plan_src": plan_src, "plan_error": plan_error,
                }

                self.ui.post(lambda res=result: self.show_result(res))

            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"

                self.ui.post(lambda msg=message: self.show_error(msg))

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    # ==================================================
    # HIEN THI KET QUA
    # ==================================================
    def show_result(self, result):
        if not self.window.winfo_exists():
            return

        self.busy = False
        self.search_button.configure(state="normal")
        self.locate_button.configure(state="normal")

        self.origin = result["origin"]
        self.destination = result["destination"]
        self.route = result["route"]
        self.places = result["places"]

        self.place_list.delete(0, tk.END)

        for place in self.places:
            label = CATEGORY_LABELS.get(
                place["kind"],
                place["kind"],
            )

            suffix = (
                f" — {place['address']}"
                if place.get("address")
                else ""
            )

            self.place_list.insert(
                tk.END,
                f"[{label}] {place['name']}{suffix}",
            )

        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()

        self.map_widget.set_path(
            self.route["points"],
            color="#1769aa",
            width=5,
        )

        self.map_widget.set_marker(
            self.origin["lat"],
            self.origin["lon"],
            text="Điểm đi",
        )

        self.map_widget.set_marker(
            self.destination["lat"],
            self.destination["lon"],
            text="Điểm đến",
        )

        marker_colors = {
            "fuel": "orange",
            "food": "blue",
            "medical": "red",
            "rest": "purple",
        }

        for place in self.places:
            color = marker_colors.get(
                place["kind"], "green"
            )

            self.map_widget.set_marker(
                place["lat"],
                place["lon"],
                text=(
                    f"{CATEGORY_LABELS.get(place['kind'], place['kind'])}: "
                    f"{place['name']}"
                ),
                marker_color_circle=color,
                marker_color_outside=color,
            )

        self._render_plan(result)

        points = self.route["points"]

        if points:
            mid = points[len(points) // 2]
            self.map_widget.set_position(mid[0], mid[1])

        distance_km = self.route["distance"] / 1000
        duration = format_duration(
            self.route["duration"]
        )

        zoom = (
            7 if distance_km > 250
            else 9 if distance_km > 70
            else 11
        )
        self.map_widget.set_zoom(zoom)

        self.summary.set(
            f"Khoảng cách: {distance_km:.1f} km | "
            f"Thời gian ước tính: {duration} | "
            f"Tìm thấy {len(self.places)} địa điểm.\n"
            "Thời gian không phản ánh ùn tắc trực tiếp."
        )

        warning = result.get("places_warning")

        if warning:
            self.status.set(
                "Đã vẽ tuyến đường; hiện chưa tải được địa điểm gợi ý. "
                "Bạn có thể thử lại sau."
            )
        else:
            src = poi_provider.last_status().get("provider", "")
            self.status.set(
                f"Đã cập nhật bản đồ, tìm thấy {len(self.places)} địa điểm"
                + (f" (nguồn: {src})." if src else ".")
            )

    # ==================================================
    # XU LY LOI TIM LO TRINH
    # ==================================================
    def show_error(self, message):
        if not self.window.winfo_exists():
            return

        self.busy = False
        self.search_button.configure(state="normal")
        self.locate_button.configure(state="normal")

        self.status.set("Không thể hoàn thành yêu cầu.")

        messagebox.showerror(
            "Lỗi bản đồ",
            message
            + "\n\nKiểm tra Internet hoặc thử lại sau.",
            parent=self.window.winfo_toplevel(),
        )

    # ==================================================
    # XOA DIEM GOI Y
    # ==================================================
    # ==================================================
    # KE HOACH DI CHUYEN AN TOAN
    # ==================================================
    @staticmethod
    def _parse_depart(text):
        import datetime as _dt
        text = (text or "").strip()
        now = _dt.datetime.now()
        if not text:
            return now
        try:
            h, m = [int(x) for x in text.replace("h", ":").split(":")[:2]]
            if not (0 <= h < 24 and 0 <= m < 60):
                return None
        except ValueError:
            return None
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if t < now - _dt.timedelta(minutes=5):      # gio da qua -> ngay mai
            t += _dt.timedelta(days=1)
        return t

    # ----- Agent AI -----
    def _agent_write(self, text, tag="bot"):
        tb = self.agent_log
        tb.configure(state="normal")
        tb._textbox.insert("end", text, tag)
        tb.configure(state="disabled")
        tb._textbox.see("end")

    def _ensure_agent(self):
        if self.agent is None:
            from trip_agent import TripAgent
            self.agent = TripAgent(
                get_position=self.position_getter, get_rag=self.rag_getter,
                trip_hooks={
                    "start": lambda r, p, o, d: self._call_on_ui(lambda: self._start_trip_with(r, p, o, d)),
                    "status": lambda: (self.trip_hooks.get("status") or (lambda: ""))(),
                    "has_plan": lambda: bool(self.plan),
                })
        return self.agent

    def _call_on_ui(self, fn):
        """Goi tu thread agent: chay fn tren main thread va doi ket qua (toi da 5 s)."""
        import threading as _th
        done = _th.Event()
        def run():
            try:
                fn()
            finally:
                done.set()
        self.ui.post(run)
        done.wait(5)

    def agent_send(self):
        if self.agent_busy:
            return
        text = self.agent_entry.get().strip()
        if not text:
            return
        self.agent_entry.delete(0, tk.END)
        self.agent_busy = True
        self.agent_send_btn.configure(state="disabled")
        self.agent_status.set("Agent đang xử lý…")
        self._agent_write("\nBạn: " + text + "\n", "me")
        TOOL_VI = {"get_current_time": "xem giờ", "get_current_position": "lấy vị trí",
                   "plan_route": "tìm tuyến đường", "find_places": "tìm địa điểm",
                   "build_safe_plan": "lập kế hoạch nghỉ", "get_safety_guidance": "tra tài liệu an toàn"}

        def on_step(step):
            label = TOOL_VI.get(step["tool"], step["tool"]) + ("" if step["ok"] else " (lỗi)")
            self.ui.post(lambda l=label: (self.agent_status.set("Agent: " + l + "…"),
                                          self._agent_write("  ⚙ " + l + "\n", "tool")))

        def worker():
            try:
                agent = self._ensure_agent()
                out = agent.chat(text, on_step=on_step)
                snap = agent.snapshot() if out["source"] in ("agent", "fallback") else None
            except Exception as exc:
                out, snap = {"text": f"Agent lỗi: {type(exc).__name__}: {exc}", "source": "error"}, None
            self.ui.post(lambda: self._agent_done(out, snap))

        threading.Thread(target=worker, daemon=True).start()

    def _agent_done(self, out, snap):
        self.agent_busy = False
        self.agent_send_btn.configure(state="normal")
        self.agent_status.set("")
        self._agent_write("Agent: " + out["text"] + "\n", "bot")
        if snap is not None and snap is not getattr(self, "_agent_last_snap", None):
            key = (id(snap["route"]), id(snap.get("plan")))
            if key != getattr(self, "_agent_last_key", None):
                self._agent_last_key = key
                self.show_result(snap)               # vẽ tuyến + kế hoạch lên bản đồ
                try:
                    self._tabs.set("Agent AI")
                except Exception:
                    pass

    # ----- Theo doi hanh trinh -----
    def toggle_trip(self):
        h = self.trip_hooks
        if not h:
            return
        if getattr(self, "trip_on", False):
            h["stop"]()
            self.trip_on = False
            self.trip_btn.configure(text="Bắt đầu hành trình")
            return
        if not (self.route and self.plan):
            return
        h["start"](self.route, self.plan, self.origin, self.destination)
        self.trip_on = True
        self.trip_btn.configure(text="Kết thúc hành trình")

    def _start_trip_with(self, route, plan, origin, destination):
        """Chay tren main thread (agent goi qua _call_on_ui)."""
        if not self.trip_hooks or getattr(self, "trip_on", False):
            return
        self.trip_hooks["start"](route, plan, origin, destination)
        self.trip_on = True
        self.trip_btn.configure(text="Kết thúc hành trình")

    def _plan_set(self, text):
        self.plan_box.configure(state="normal")
        self.plan_box.delete("1.0", "end")
        self.plan_box.insert("1.0", text)
        self.plan_box.configure(state="disabled")

    def _render_plan(self, result):
        self.plan = result.get("plan")
        try:
            self.trip_btn.configure(state="normal" if self.plan else "disabled")
        except Exception:
            pass
        if result.get("plan_error"):
            self._plan_set("Chưa lập được kế hoạch: " + result["plan_error"] +
                           "\nTuyến đường vẫn hiển thị bình thường.")
            return
        if not self.plan:
            self._plan_set("Kế hoạch đang tắt. Bật «Lập kế hoạch nghỉ» rồi tìm lại lộ trình.")
            return
        import trip_planner
        icons = {"rest": "purple", "food": "purple", "unknown": "gray", "fuel": "orange"}
        for i, st in enumerate(self.plan["stops"], 1):
            color = icons.get(st["kind"], "purple")
            self.map_widget.set_marker(
                st["lat"], st["lon"],
                text=f"Dừng {i}: {st['name']}",
                marker_color_circle=color, marker_color_outside=color)
        lines = trip_planner.plan_lines(self.plan)
        src = "Gemini" if result.get("plan_src") == "gemini" else "mẫu có sẵn"
        text = (result.get("plan_text") or "") + f"\n(lời tóm tắt: {src}; số liệu do hệ thống tính)\n\n"
        text += "\n".join(lines)
        text += ("\n\nƯớc tính từ tuyến đường, đã cộng dự phòng; chưa tính ùn tắc. "
                 "Hãy dừng nghỉ ngay khi thấy mệt.")
        self._plan_set(text)

    def clear_places(self):
        self.places = []
        self.place_list.delete(0, tk.END)

        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()

        if self.origin:
            self.map_widget.set_marker(
                self.origin["lat"],
                self.origin["lon"],
                text="Điểm đi",
            )

        if self.destination:
            self.map_widget.set_marker(
                self.destination["lat"],
                self.destination["lon"],
                text="Điểm đến",
            )

        if self.route:
            self.map_widget.set_path(
                self.route["points"],
                color="#1769aa",
                width=5,
            )

        self.status.set("Đã xóa các điểm gợi ý.")

    # ==================================================
    # DI CHUYEN DEN DIA DIEM DUOC CHON
    # ==================================================
    def focus_selected_place(self, _event=None):
        selection = self.place_list.curselection()

        if not selection:
            return

        index = selection[0]

        if index >= len(self.places):
            return

        place = self.places[index]

        self.map_widget.set_position(
            place["lat"],
            place["lon"],
        )
        self.map_widget.set_zoom(15)


_planner = None


def open_route_planner(parent):
    """Mo ban do trong cua so rieng (hoac tra ve ban do dang co)."""
    global _planner
    try:
        if _planner is not None and _planner.window.winfo_exists():
            top = _planner.window.winfo_toplevel()
            top.deiconify()
            top.lift()
            return _planner
    except Exception:
        pass
    _planner = RoutePlanner(parent)
    return _planner