"""Tien ich hinh hoc cho tuyen duong (khong phu thuoc mang / giao dien)."""
import math


def haversine_m(lat1, lon1, lat2, lon2):
    """Khoang cach (met) giua hai toa do."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def cumulative_m(points):
    cum = [0.0]
    for a, b in zip(points, points[1:]):
        cum.append(cum[-1] + haversine_m(a[0], a[1], b[0], b[1]))
    return cum


def sample_route(points, step_m=4000, max_points=40):
    """Lay cac diem doc tuyen cach nhau ~step_m met; giu diem dau va cuoi."""
    if not points:
        return []
    if len(points) == 1:
        return list(points)

    picked = [points[0]]
    acc = 0.0
    for prev, cur in zip(points, points[1:]):
        acc += haversine_m(prev[0], prev[1], cur[0], cur[1])
        if acc >= step_m:
            picked.append(cur)
            acc = 0.0
    if picked[-1] != points[-1]:
        picked.append(points[-1])

    if len(picked) > max_points:
        stride = len(picked) / float(max_points)
        picked = [picked[int(i * stride)] for i in range(max_points)]
        if picked[-1] != points[-1]:
            picked.append(points[-1])
    return picked


def sample_route_n(points, n):
    """Lay dung n diem cach deu nhau theo QUANG DUONG (gom diem dau va cuoi)."""
    if not points:
        return []
    if n <= 1 or len(points) == 1:
        return [points[0]]
    cum = cumulative_m(points)
    total = cum[-1]
    if total <= 0:
        return [points[0]]
    out, j = [], 0
    for i in range(n):
        target = total * i / (n - 1)
        while j < len(cum) - 1 and cum[j] < target:
            j += 1
        out.append(points[j])
    # bo trung lap
    uniq = []
    for p in out:
        if not uniq or p != uniq[-1]:
            uniq.append(p)
    return uniq


def places_ahead_on_route(route_points, cur_lat, cur_lon, places,
                          max_ahead_m=20000, max_offset_m=1500):
    """
    Loc cac dia diem nam PHIA TRUOC vi tri hien tai doc theo tuyen.
    Moi dia diem duoc them 'dist_m' = khoang cach doc tuyen (met).
    """
    if not route_points or not places:
        return []

    pts = route_points
    if len(pts) > 1500:
        stride = len(pts) / 1500.0
        pts = [pts[int(i * stride)] for i in range(1500)]

    cum = cumulative_m(pts)

    def nearest(lat, lon):
        best_i, best_d = 0, float("inf")
        for i, (plat, plon) in enumerate(pts):
            d = haversine_m(lat, lon, plat, plon)
            if d < best_d:
                best_i, best_d = i, d
        return best_i, best_d

    cur_i, cur_off = nearest(cur_lat, cur_lon)
    if cur_off > 3000:
        return []          # lech xa tuyen: khong the noi "phia truoc"

    result = []
    for place in places:
        idx, off = nearest(place["lat"], place["lon"])
        ahead = cum[idx] - cum[cur_i]
        if off <= max_offset_m and 0 < ahead <= max_ahead_m:
            item = dict(place)
            item["dist_m"] = ahead
            result.append(item)

    result.sort(key=lambda p: p["dist_m"])
    return result