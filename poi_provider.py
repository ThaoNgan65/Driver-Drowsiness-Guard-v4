"""
poi_provider.py - Tim dia diem (tram xang, quan an, y te, diem dung chan) quanh mot vi tri.

Nguon du lieu (chon theo API key co trong .env / bien moi truong):
  1. Geoapify Places  (GEOAPIFY_API_KEY)      - goi mien phi 3000 request/ngay
  2. Google Places New (GOOGLE_PLACES_API_KEY) - phu song tot nhat, co tinh phi
  3. Overpass (OpenStreetMap)                  - chi la phuong an cuoi: khi chua co key
                                                 hoac nguon chinh loi (tat bang POI_OVERPASS_FALLBACK=0)
Ep nguon: POI_PROVIDER=geoapify | google | overpass

Moi nhom (kind) = 1 request, chay song song. Ket qua duoc cache ra dia (poi_cache.json),
cache cu van duoc dung khi mat mang.
"""
import os
import json
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutTimeout

import requests

import settings
from geo_utils import haversine_m, sample_route_n

log = logging.getLogger("drowsiness")

HEADERS = {"User-Agent": "DriverSafetyApp/1.0"}
KINDS = ("rest", "food", "fuel", "medical")
KIND_DEFAULT_NAME = {
    "rest": "Điểm dừng chân",
    "food": "Quán ăn / cà phê",
    "fuel": "Trạm xăng",
    "medical": "Cơ sở y tế",
}
REQ_TIMEOUT = (3, 6)         # (connect, read) giay
BATCH_DEADLINE = 12          # toi da cho 1 dot truy van song song
PER_KIND_LIMIT = 8

# ---- Geoapify (da doi chieu tai lieu Places API) ----
GEOAPIFY_URL = "https://api.geoapify.com/v2/places"
GEOAPIFY_CATS = {
    "fuel": "service.vehicle.fuel",
    "food": "catering.restaurant,catering.cafe",
    "medical": "healthcare.hospital,healthcare.pharmacy",
    "rest": "parking,amenity.toilet",
}

# ---- Google Places API (New) - ten type da doi chieu Table A ----
GOOGLE_URL = "https://places.googleapis.com/v1/places:searchNearby"
GOOGLE_TYPES = {
    "fuel": ["gas_station"],
    "food": ["restaurant", "cafe", "coffee_shop"],
    "medical": ["hospital", "pharmacy", "drugstore", "doctor"],
    "rest": ["rest_stop", "parking", "public_bathroom"],
}
GOOGLE_FIELDS = "places.displayName,places.location,places.formattedAddress"

# ---- Overpass (du phong) ----
OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_AMENITY = {
    "fuel": "fuel",
    "food": "restaurant|cafe|fast_food",
    "medical": "hospital|clinic|pharmacy",
    "rest": "parking|toilets",
}
AMENITY_TO_KIND = {
    "fuel": "fuel", "restaurant": "food", "cafe": "food", "fast_food": "food",
    "hospital": "medical", "clinic": "medical", "pharmacy": "medical",
    "parking": "rest", "toilets": "rest",
}


class ProviderError(RuntimeError):
    pass


_state = {"provider": "", "error": "", "ts": 0.0}


def last_status():
    """Cho UI hien thi: nguon dang dung va loi gan nhat (neu co)."""
    return dict(_state)


def configured_provider():
    forced = settings.get("POI_PROVIDER").lower()
    if forced in ("geoapify", "google", "overpass"):
        return forced
    if settings.get("GEOAPIFY_API_KEY"):
        return "geoapify"
    if settings.get("GOOGLE_PLACES_API_KEY"):
        return "google"
    return "overpass"


def provider_chain():
    primary = configured_provider()
    chain = [primary]
    if primary != "overpass" and settings.get("POI_OVERPASS_FALLBACK", "1") != "0":
        chain.append("overpass")
    return chain


# ---------------------------------------------------------------- nguon
def _clean_name(name, kind):
    name = (name or "").strip()
    return name if name else KIND_DEFAULT_NAME.get(kind, "Địa điểm")


def _geoapify_kind(kind, lat, lon, radius, limit):
    key = settings.get("GEOAPIFY_API_KEY")
    if not key:
        raise ProviderError("Thiếu GEOAPIFY_API_KEY")
    resp = requests.get(
        GEOAPIFY_URL,
        params={
            "categories": GEOAPIFY_CATS[kind],
            "filter": f"circle:{lon:.6f},{lat:.6f},{int(radius)}",
            "bias": f"proximity:{lon:.6f},{lat:.6f}",
            "limit": limit,
            "apiKey": key,
        },
        headers=HEADERS,
        timeout=REQ_TIMEOUT,
    )
    if resp.status_code in (401, 403):
        raise ProviderError("Geoapify từ chối API key (HTTP %d)" % resp.status_code)
    if resp.status_code == 429:
        raise ProviderError("Geoapify hết hạn mức (HTTP 429)")
    resp.raise_for_status()
    out = []
    for feat in resp.json().get("features", []):
        props = feat.get("properties", {})
        plat, plon = props.get("lat"), props.get("lon")
        if plat is None or plon is None:
            coords = (feat.get("geometry") or {}).get("coordinates") or []
            if len(coords) >= 2:
                plon, plat = coords[0], coords[1]
        if plat is None or plon is None:
            continue
        out.append({
            "name": _clean_name(props.get("name"), kind),
            "lat": float(plat), "lon": float(plon), "kind": kind,
            "address": props.get("address_line2") or props.get("formatted") or "",
        })
    return out


def _google_kind(kind, lat, lon, radius, limit):
    key = settings.get("GOOGLE_PLACES_API_KEY")
    if not key:
        raise ProviderError("Thiếu GOOGLE_PLACES_API_KEY")
    resp = requests.post(
        GOOGLE_URL,
        json={
            "includedTypes": GOOGLE_TYPES[kind],
            "maxResultCount": min(int(limit), 20),
            "rankPreference": "DISTANCE",
            "languageCode": "vi",
            "locationRestriction": {"circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": float(min(radius, 50000)),
            }},
        },
        headers={
            **HEADERS,
            "Content-Type": "application/json",
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": GOOGLE_FIELDS,
        },
        timeout=REQ_TIMEOUT,
    )
    if resp.status_code in (400, 401, 403):
        try:
            msg = resp.json().get("error", {}).get("message", "")
        except Exception:
            msg = ""
        raise ProviderError(f"Google Places lỗi HTTP {resp.status_code}: {msg[:120]}")
    resp.raise_for_status()
    out = []
    for p in resp.json().get("places", []):
        loc = p.get("location") or {}
        if "latitude" not in loc or "longitude" not in loc:
            continue
        out.append({
            "name": _clean_name((p.get("displayName") or {}).get("text"), kind),
            "lat": float(loc["latitude"]), "lon": float(loc["longitude"]),
            "kind": kind, "address": p.get("formattedAddress", ""),
        })
    return out


def _overpass_one(endpoint, query):
    resp = requests.post(endpoint, data={"data": query}, headers=HEADERS, timeout=(3, 10))
    if resp.status_code != 200:
        raise ProviderError(f"HTTP {resp.status_code}")
    data = resp.json()
    if "elements" not in data:
        raise ProviderError("phản hồi không có 'elements'")
    return data


def _overpass_query(query):
    errors = []
    pool = ThreadPoolExecutor(max_workers=len(OVERPASS_ENDPOINTS))
    futs = {pool.submit(_overpass_one, ep, query): ep for ep in OVERPASS_ENDPOINTS}
    try:
        for fut in as_completed(futs, timeout=BATCH_DEADLINE):
            try:
                return fut.result()
            except Exception as exc:
                errors.append(f"{futs[fut]}: {exc}")
    except FutTimeout:
        errors.append("hết thời gian chờ")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    raise ProviderError("Overpass lỗi: " + " | ".join(errors))


def _overpass_parse(data, kinds):
    out, seen = [], set()
    for item in data.get("elements", []):
        tags = item.get("tags", {})
        lat, lon = item.get("lat"), item.get("lon")
        if lat is None or lon is None:
            c = item.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        kind = AMENITY_TO_KIND.get(tags.get("amenity", ""))
        if kind is None or kind not in kinds:
            continue
        key = (round(float(lat), 5), round(float(lon), 5), kind)
        if key in seen:
            continue
        seen.add(key)
        addr = tags.get("addr:full") or " ".join(filter(None, [
            tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")]))
        out.append({
            "name": _clean_name(tags.get("name") or tags.get("name:vi") or tags.get("brand"), kind),
            "lat": float(lat), "lon": float(lon), "kind": kind, "address": addr,
        })
    return out


def _overpass_around(kinds, around_spec):
    parts = []
    for k in kinds:
        parts.append(f'nwr["amenity"~"{OVERPASS_AMENITY[k]}"]({around_spec});')
    query = "[out:json][timeout:10];(" + "".join(parts) + ");out center tags 400;"
    return _overpass_parse(_overpass_query(query), kinds)


# ---------------------------------------------------------------- dieu phoi
def _parallel(jobs, deadline=BATCH_DEADLINE):
    """jobs: list[(label, fn, args)] -> (results_flat, errors)."""
    results, errors = [], []
    if not jobs:
        return results, errors
    pool = ThreadPoolExecutor(max_workers=min(8, len(jobs)))
    futs = {pool.submit(fn, *args): label for label, fn, args in jobs}
    try:
        for fut in as_completed(futs, timeout=deadline):
            try:
                results.extend(fut.result())
            except Exception as exc:
                errors.append(f"{futs[fut]}: {exc}")
    except FutTimeout:
        errors.append("hết thời gian chờ")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results, errors


def _dedupe(places):
    seen, out = set(), []
    for p in places:
        key = (round(p["lat"], 5), round(p["lon"], 5), p["kind"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _fetch_circle(provider, lat, lon, radius, kinds):
    """Mot vong tron quanh (lat,lon). Nem ProviderError neu TAT CA that bai."""
    if provider == "overpass":
        spec = f"around:{int(radius)},{lat:.6f},{lon:.6f}"
        return _overpass_around(kinds, spec)
    fn = _geoapify_kind if provider == "geoapify" else _google_kind
    jobs = [(k, fn, (k, lat, lon, radius, PER_KIND_LIMIT)) for k in kinds]
    results, errors = _parallel(jobs)
    if errors and not results:
        raise ProviderError("; ".join(errors))
    if errors:
        log.warning("POI %s: mot so nhom loi: %s", provider, "; ".join(errors))
    return _dedupe(results)


def _fetch_with_fallback(lat, lon, radius, kinds):
    errors = []
    for provider in provider_chain():
        try:
            places = _fetch_circle(provider, lat, lon, radius, kinds)
            _state.update(provider=provider, error="; ".join(errors), ts=time.time())
            return places
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            log.warning("POI nguon %s loi: %s", provider, exc)
    _state.update(provider=configured_provider(), error="; ".join(errors), ts=time.time())
    raise ProviderError("; ".join(errors))


# ---------------------------------------------------------------- cache dia
_POI_FILE = os.path.join(settings.BASE_DIR, "poi_cache.json")
_NEAR_TTL = 1800
_NEAR_REUSE_M = 1500
_NEAR_EXTRA_M = 1500
_NEAR_MAX_ENTRIES = 30
_lock = threading.Lock()
_store = None


def _load():
    global _store
    if _store is None:
        try:
            with open(_POI_FILE, encoding="utf-8") as f:
                _store = json.load(f)
            if not isinstance(_store, list):
                _store = []
        except Exception:
            _store = []
    return _store


def _save():
    try:
        tmp = _POI_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_store, f, ensure_ascii=False)
        os.replace(tmp, _POI_FILE)
    except Exception as exc:
        log.warning("Khong luu duoc poi_cache: %s", exc)


def _with_dist(places, lat, lon, radius):
    out = []
    for p in places:
        d = haversine_m(lat, lon, p["lat"], p["lon"])
        if d <= radius:
            item = dict(p)
            item["dist_m"] = d
            out.append(item)
    out.sort(key=lambda x: x["dist_m"])
    return out


def _cache_get(lat, lon, tag, radius, allow_stale=False):
    with _lock:
        for e in _load():
            if e.get("tag") != tag:
                continue
            moved = haversine_m(lat, lon, e["lat"], e["lon"])
            if moved > _NEAR_REUSE_M or e["radius"] < radius + moved:
                continue
            if allow_stale or (time.time() - e["ts"]) < _NEAR_TTL:
                return _with_dist(e["places"], lat, lon, radius)
    return None


def _cache_put(lat, lon, tag, fetch_radius, places):
    with _lock:
        store = _load()
        store[:] = [e for e in store if not (
            e.get("tag") == tag and haversine_m(lat, lon, e["lat"], e["lon"]) < 300)]
        store.append({"ts": time.time(), "tag": tag, "lat": lat, "lon": lon,
                      "radius": fetch_radius, "places": places})
        del store[:-_NEAR_MAX_ENTRIES]
        _save()


def _kinds_for(category):
    return KINDS if category in ("all", "", None) else (category,)


# ---------------------------------------------------------------- API cong khai
def find_near(lat, lon, category="all", radius=3000):
    """Dia diem quanh mot vi tri, co 'dist_m', gan -> xa. Nem ProviderError neu khong co nguon nao."""
    kinds = _kinds_for(category)
    tag = ",".join(sorted(kinds))

    cached = _cache_get(lat, lon, tag, radius)
    if cached is not None:
        return cached

    fetch_radius = radius + _NEAR_EXTRA_M
    try:
        places = _fetch_with_fallback(lat, lon, fetch_radius, kinds)
    except ProviderError:
        stale = _cache_get(lat, lon, tag, radius, allow_stale=True)
        if stale is not None:
            log.warning("POI: dung cache cu vi khong co nguon nao tra loi")
            return stale
        raise
    _cache_put(lat, lon, tag, fetch_radius, places)
    return _with_dist(places, lat, lon, radius)


_route_cache = {}


def find_along_route(points, category="all", radius=1500, max_samples=5):
    """Dia diem doc tuyen (khong co dist_m). Tra [] neu loi (de giu tuyen duong tren ban do)."""
    if not points:
        return []
    kinds = _kinds_for(category)
    samples = sample_route_n(points, max_samples)
    key = (",".join(sorted(kinds)), int(radius),
           tuple((round(a, 4), round(b, 4)) for a, b in samples))
    hit = _route_cache.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]

    jobs = [(f"{i}", _fetch_with_fallback, (lat, lon, radius, kinds))
            for i, (lat, lon) in enumerate(samples)]
    results, errors = _parallel(jobs, deadline=BATCH_DEADLINE + 6)
    if errors and not results:
        log.warning("POI doc tuyen loi: %s", "; ".join(errors))
        return []
    places = _dedupe(results)
    _route_cache[key] = (time.time(), places)
    return places