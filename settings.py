"""
Cau hinh ung dung. Doc bien moi truong, va file .env canh code (neu co).

Vi du .env:
    GEMINI_API_KEY=...
    GEOAPIFY_API_KEY=...        # nguon dia diem mac dinh (goi mien phi 3000 request/ngay)
    GOOGLE_PLACES_API_KEY=...   # tuy chon (co tinh phi)
    POI_PROVIDER=geoapify       # geoapify | google | overpass (de trong = tu chon theo key)
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env(path=None):
    path = path or os.path.join(BASE_DIR, ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Khong doc duoc .env: {exc}")


def get(name, default=""):
    return os.environ.get(name, default).strip()


load_env()