"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
location_gps_geofence.py
────────────────────────────────────────────────────────────────
MODULE A: GPS Geofencing Zone Classifier (Primary Method)

ทำงาน:
  - รับ lat, lon จาก GPS มือถือ
  - ตรวจว่าอยู่ใน Zone ไหน (radius geofence)
  - Output: zone_name, confidence, distance_to_center

ทดสอบโดยไม่ต้องมี GPS จริง:
  python location_gps_geofence.py --simulate

ใช้งาน (real GPS):
  python location_gps_geofence.py --lat 13.6512 --lon 100.4930
"""

import argparse
import math
import json
from dataclasses import dataclass, field
from typing import Optional


# ─── Zone Definition ──────────────────────────────────────────
@dataclass
class GeoZone:
    """นิยาม 1 zone โดยใช้ center point + radius"""
    zone_id   : str
    zone_name : str           # ชื่อห้อง/พื้นที่
    lat       : float         # latitude ของจุดกลาง
    lon       : float         # longitude ของจุดกลาง
    radius_m  : float         # รัศมี (เมตร) ถือว่าอยู่ใน zone ถ้า distance ≤ radius
    risk_level: str = "low"   # risk level ของ zone นี้ (low/medium/high)
    tags      : list = field(default_factory=list)  # metadata


# ─── Default Zone Configuration ───────────────────────────────
# *** แก้ค่า lat/lon ให้ตรงกับสถานที่ทดสอบจริง ***
# ตัวอย่างนี้ตั้งไว้แถว KMUTT บางมด (ปรับตามจริง)
DEFAULT_ZONES = [
    GeoZone(
        zone_id="bedroom",
        zone_name="ห้องนอน",
        lat=13.6512, lon=100.4930,
        radius_m=5.0,
        risk_level="medium",   # ล้มในห้องนอน = ความเสี่ยงระดับกลาง
        tags=["indoor", "sleeping", "private"]
    ),
    GeoZone(
        zone_id="bathroom",
        zone_name="ห้องน้ำ",
        lat=13.6513, lon=100.4931,
        radius_m=3.0,
        risk_level="high",     # ห้องน้ำ = เสี่ยงสูงสุด (พื้นลื่น)
        tags=["indoor", "wet_floor", "high_risk"]
    ),
    GeoZone(
        zone_id="living_room",
        zone_name="ห้องนั่งเล่น",
        lat=13.6511, lon=100.4929,
        radius_m=8.0,
        risk_level="medium",
        tags=["indoor", "common_area"]
    ),
    GeoZone(
        zone_id="kitchen",
        zone_name="ห้องครัว",
        lat=13.6510, lon=100.4932,
        radius_m=4.0,
        risk_level="medium",
        tags=["indoor", "cooking_area"]
    ),
    GeoZone(
        zone_id="outdoor_yard",
        zone_name="ลานนอกบ้าน",
        lat=13.6509, lon=100.4928,
        radius_m=15.0,
        risk_level="low",
        tags=["outdoor"]
    ),
    GeoZone(
        zone_id="staircase",
        zone_name="บริเวณบันได",
        lat=13.6512, lon=100.4933,
        radius_m=2.5,
        risk_level="high",     # บันได = เสี่ยงสูง
        tags=["indoor", "stairs", "high_risk"]
    ),
]


# ─── Haversine Distance ───────────────────────────────────────
def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """คำนวณระยะทางระหว่าง 2 จุด GPS (เมตร) โดยใช้ Haversine formula"""
    R = 6_371_000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── GPS Geofence Classifier ──────────────────────────────────
class GPSGeofenceClassifier:
    """
    Classify GPS coordinates → Zone

    Usage:
        clf = GPSGeofenceClassifier(zones=DEFAULT_ZONES)
        result = clf.classify(lat=13.6512, lon=100.4930)
        print(result)
    """

    def __init__(self, zones: list[GeoZone] = None):
        self.zones = zones or DEFAULT_ZONES

    def classify(self, lat: float, lon: float) -> dict:
        """
        รับ lat, lon → คืน dict ของผลการจำแนก zone

        Returns:
            {
              "zone_id": "bedroom",
              "zone_name": "ห้องนอน",
              "risk_level": "medium",
              "distance_m": 1.23,
              "confidence": 0.95,
              "source": "gps",
              "all_zones": [...]  # ระยะทางไปทุก zone
            }
        """
        distances = []
        for zone in self.zones:
            dist = haversine_distance_m(lat, lon, zone.lat, zone.lon)
            distances.append((zone, dist))

        # เรียงตามระยะทาง
        distances.sort(key=lambda x: x[1])
        closest_zone, closest_dist = distances[0]

        # ถ้าอยู่ใน radius → matched
        if closest_dist <= closest_zone.radius_m:
            confidence = max(0.0, 1.0 - (closest_dist / closest_zone.radius_m) * 0.5)
            return {
                "zone_id":    closest_zone.zone_id,
                "zone_name":  closest_zone.zone_name,
                "risk_level": closest_zone.risk_level,
                "distance_m": round(closest_dist, 2),
                "confidence": round(confidence, 3),
                "source":     "gps",
                "tags":       closest_zone.tags,
                "matched":    True,
                "all_zones":  [
                    {"zone_id": z.zone_id, "dist_m": round(d, 2), "in_zone": d <= z.radius_m}
                    for z, d in distances
                ]
            }
        else:
            # ไม่ได้อยู่ใน zone ไหนเลย → unknown
            # confidence ลดตามระยะห่าง
            confidence = max(0.0, 1.0 - (closest_dist / (closest_zone.radius_m * 3)))
            return {
                "zone_id":    "unknown",
                "zone_name":  "พื้นที่ไม่ระบุ",
                "risk_level": "low",
                "distance_m": round(closest_dist, 2),
                "confidence": round(confidence, 3),
                "source":     "gps",
                "tags":       ["unknown"],
                "matched":    False,
                "nearest_zone": closest_zone.zone_id,
                "all_zones":  [
                    {"zone_id": z.zone_id, "dist_m": round(d, 2), "in_zone": d <= z.radius_m}
                    for z, d in distances
                ]
            }

    def load_zones_from_json(self, path: str):
        """โหลด zone config จาก JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.zones = [GeoZone(**z) for z in data["zones"]]
        print(f"[GPS] Loaded {len(self.zones)} zones from {path}")

    def export_zones_to_json(self, path: str):
        """Export zone config ออกเป็น JSON (สำหรับแชร์กับทีม)"""
        data = {
            "zones": [
                {
                    "zone_id":    z.zone_id,
                    "zone_name":  z.zone_name,
                    "lat":        z.lat,
                    "lon":        z.lon,
                    "radius_m":   z.radius_m,
                    "risk_level": z.risk_level,
                    "tags":       z.tags,
                }
                for z in self.zones
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[GPS] Exported {len(self.zones)} zones to {path}")


# ─── Simulate Test ────────────────────────────────────────────
def run_simulate_test():
    """
    ทดสอบ GPS Geofence ด้วยพิกัดจำลอง
    ทดสอบทุก scenario: ใน zone, นอก zone, บน boundary
    """
    clf = GPSGeofenceClassifier()

    print("\n" + "="*68)
    print("  MODULE A: GPS Geofencing — Simulate Test")
    print("="*68)
    print(f"  Zones defined: {len(clf.zones)}")
    for z in clf.zones:
        zname_safe = z.zone_name.encode('ascii','replace').decode('ascii')
        print(f"    [{z.zone_id:<15}] {zname_safe:<15} center=({z.lat},{z.lon}) r={z.radius_m}m risk={z.risk_level}")
    print("="*68)

    # Test cases: (description, lat, lon, expected_zone)
    test_cases = [
        # ── อยู่ใน zone ────────────────────────────────────────────────
        ("อยู่กลางห้องนอน (exact center)",     13.6512,  100.4930,  "bedroom"),
        ("อยู่ห้องน้ำ (exact center)",          13.6513,  100.4931,  "bathroom"),
        ("อยู่ห้องนั่งเล่น (near center)",     13.65111, 100.49292, "living_room"),
        ("อยู่บันได (exact center)",            13.6512,  100.4933,  "staircase"),
        ("อยู่ลานกลางแจ้ง",                    13.6509,  100.4928,  "outdoor_yard"),
        # ── อยู่บน boundary ────────────────────────────────────────────
        ("ขอบห้องนอน (~4.9m จากศูนย์)",       13.6512,  100.49344, "bedroom"),
        ("ขอบห้องน้ำ (~2.9m จากศูนย์)",       13.65132, 100.4931,  "bathroom"),
        # ── นอก zone ──────────────────────────────────────────────────
        ("นอกทุก zone (ห่างมาก)",              13.6520,  100.4950,  "unknown"),
        ("ระหว่าง 2 zones (corridor?)",        13.65115, 100.49305, None),  # อาจ zone ไหนก็ได้
    ]

    print(f"\n{'No.':<4} {'คำอธิบาย':<40} {'Expected':<15} {'Got':<15} {'Conf':<6} {'Dist(m)':<8} {'OK?'}")
    print("-"*105)

    passed = 0
    for i, (desc, lat, lon, expected) in enumerate(test_cases, 1):
        result = clf.classify(lat, lon)
        got    = result["zone_id"]
        conf   = result["confidence"]
        dist   = result["distance_m"]

        if expected is None:
            ok = "--"  # ไม่กำหนด expected
        elif got == expected:
            ok = "PASS"
            passed += 1
        else:
            ok = "FAIL"

        exp_str = expected or "any"
        desc_safe = desc.encode('ascii','replace').decode('ascii')
        print(f"{i:<4} {desc_safe:<40} {exp_str:<15} {got:<15} {conf:<6.3f} {dist:<8.2f} {ok}")

    determined = sum(1 for _, _, _, e in test_cases if e is not None)
    print("-"*105)
    print(f"\nผล: {passed}/{determined} test cases ผ่าน (ไม่นับ 'any')")
    print(f"GPS Geofence Accuracy: {passed/determined*100:.1f}%\n")

    # ── Export zone config ────────────────────────────────────────────
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out_json = os.path.join(here, "location_zones_config.json")
    clf.export_zones_to_json(out_json)
    print(f"Zone config saved → {out_json}")
    print("  (แก้ไข lat/lon ในไฟล์ JSON ให้ตรงกับสถานที่จริงแล้ว import กลับมาใช้ได้)")

    return passed / determined


# ─── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPS Geofence Zone Classifier")
    parser.add_argument("--lat",      type=float, default=None)
    parser.add_argument("--lon",      type=float, default=None)
    parser.add_argument("--simulate", action="store_true", help="รัน simulate test")
    parser.add_argument("--zones",    type=str,   default=None, help="Path to zones JSON config")
    args = parser.parse_args()

    clf = GPSGeofenceClassifier()
    if args.zones:
        clf.load_zones_from_json(args.zones)

    if args.simulate or (args.lat is None):
        run_simulate_test()
    else:
        result = clf.classify(args.lat, args.lon)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
