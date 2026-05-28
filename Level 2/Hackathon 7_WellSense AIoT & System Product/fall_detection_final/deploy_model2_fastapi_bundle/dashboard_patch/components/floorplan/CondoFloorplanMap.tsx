"use client";

import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { Bath, BedDouble, ChefHat, DoorOpen, Sofa, Trees, Waves } from "lucide-react";
import { FootstepTrail, sampleMovementPath } from "@/components/floorplan/FootstepTrail";
import { HeatmapOverlay } from "@/components/floorplan/HeatmapOverlay";
import { NearFallMarker } from "@/components/floorplan/NearFallMarker";
import { riskZones, rooms } from "@/data/floorplan";
import { RoomName } from "@/lib/types";
import { cn, riskColor } from "@/lib/utils";
import { useMonitoringStore } from "@/store/monitoring-store";

const roomIcons: Record<RoomName, typeof BedDouble> = {
  Bedroom: BedDouble,
  Bathroom: Bath,
  Kitchen: ChefHat,
  "Living Room": Sofa,
  Hallway: DoorOpen,
  Balcony: Trees,
};

const roomThai: Record<RoomName, string> = {
  Bedroom: "ห้องนอน",
  Bathroom: "ห้องน้ำ",
  Kitchen: "ห้องครัว",
  "Living Room": "ห้องนั่งเล่น",
  Hallway: "ทางเดิน",
  Balcony: "ระเบียง",
};

const roomGuidance: Record<
  RoomName,
  {
    focus: string;
    walking: string;
    activities: Array<{ code: string; label: string }>;
    cautions: string[];
  }
> = {
  Bedroom: {
    focus: "ช่วงลุกจากเตียงและยืนทรงตัวตอนเช้า",
    walking: "มักเป็นการเดินช้า ก้าวสั้น และมีจังหวะหยุดก่อนออกจากห้อง",
    activities: [
      { code: "D01", label: "Walking slowly" },
      { code: "D07", label: "Sit slowly / stand up slowly" },
      { code: "D12", label: "Lie slowly" },
    ],
    cautions: ["ระวังข้างเตียง", "เปิดไฟก่อนลุก", "เก็บของบนพื้นให้โล่ง"],
  },
  Bathroom: {
    focus: "พื้นเปียก การหมุนตัว และการเข้าออกประตู",
    walking: "มักเดินช้า ก้าวระวัง และมีการหมุนตัวแคบใกล้สุขภัณฑ์",
    activities: [
      { code: "D01", label: "Walking slowly" },
      { code: "D18", label: "Stumble" },
      { code: "F04", label: "Forward trip fall risk" },
    ],
    cautions: ["ระวังพื้นลื่น", "จับราวพยุงก่อนหมุนตัว", "เช็ดพื้นเปียกทันที"],
  },
  Hallway: {
    focus: "ทางแคบ จุดเลี้ยว และการเดินตอนกลางคืน",
    walking: "พบการเดินช้า สลับหยุด และอาจมีการเสียจังหวะเมื่อเลี้ยว",
    activities: [
      { code: "D01", label: "Walking slowly" },
      { code: "D02", label: "Walking quickly" },
      { code: "D18", label: "Stumble" },
    ],
    cautions: ["ระวังมุมเลี้ยว", "เปิดไฟทางเดิน", "อย่าวางของกีดขวาง"],
  },
  Kitchen: {
    focus: "ขอบเคาน์เตอร์ การเอื้อมหยิบของ และพื้นต่างระดับเล็กน้อย",
    walking: "มักเดินช้าใกล้เคาน์เตอร์และมีการหยุดยืนเพื่อหยิบของ",
    activities: [
      { code: "D01", label: "Walking slowly" },
      { code: "D16", label: "Bend without knees" },
      { code: "D18", label: "Stumble" },
    ],
    cautions: ["ระวังขอบเคาน์เตอร์", "หลีกเลี่ยงการเอื้อมไกล", "เช็ดคราบน้ำ/น้ำมัน"],
  },
  "Living Room": {
    focus: "การลุกนั่งจากโซฟาและการเดินอ้อมเฟอร์นิเจอร์",
    walking: "มักมีการเดินช้าแล้วหยุดนั่ง หรือเปลี่ยนท่าลุกนั่งซ้ำ",
    activities: [
      { code: "D07", label: "Sit slowly" },
      { code: "D08", label: "Sit quickly" },
      { code: "D11", label: "Try stand then collapse to chair" },
    ],
    cautions: ["ระวังขอบโต๊ะ", "ลุกช้า ๆ จากโซฟา", "วางรีโมต/ของใช้ใกล้มือ"],
  },
  Balcony: {
    focus: "ธรณีประตู แสงต่างระดับ และพื้นภายนอก",
    walking: "มักมีการชะลอก่อนก้าวข้ามธรณี และก้าวสั้นบริเวณทางออก",
    activities: [
      { code: "D01", label: "Walking slowly" },
      { code: "D05", label: "Upstairs/downstairs slowly" },
      { code: "D18", label: "Stumble" },
    ],
    cautions: ["ระวังธรณีประตู", "เช็กพื้นเปียก", "ใช้รองเท้ากันลื่น"],
  },
};

export function CondoFloorplanMap({
  className,
  detailed,
  compact,
  playbackIndex,
}: {
  className?: string;
  detailed?: boolean;
  compact?: boolean;
  playbackIndex?: number;
}) {
  const { readings, heatPoints, livePosition, roomRisks, nightMode } =
    useMonitoringStore();
  const [hoveredRoom, setHoveredRoom] = useState<RoomName | null>(null);
  const [selectedRoom, setSelectedRoom] = useState<RoomName | null>(null);
  const activeReading =
    typeof playbackIndex === "number"
      ? readings[Math.min(playbackIndex, readings.length - 1)] ?? livePosition
      : livePosition;
  const nearFalls = readings.filter((reading) => reading.near_fall).slice(compact ? -2 : -4);
  const visibleHeatPoints = useMemo(
    () =>
      [...riskZones, ...heatPoints]
        .filter((point) => point.intensity >= 36)
        .filter((point, index, all) => index === all.findIndex((item) => item.id === point.id))
        .slice(compact ? -14 : -22),
    [compact, heatPoints],
  );

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_18px_55px_rgba(15,23,42,0.12)]",
        nightMode && "bg-slate-50",
        className,
      )}
    >
      <svg
        viewBox="0 0 760 500"
        className="h-full min-h-[inherit] w-full"
        role="img"
        aria-label="แผนที่บ้านพร้อม heatmap ความเสี่ยงการล้ม"
      >
        <defs>
          <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="8" stdDeviation="8" floodColor="#64748b" floodOpacity=".14" />
          </filter>
          <pattern id="floor-grid" width="16" height="16" patternUnits="userSpaceOnUse">
            <path d="M16 0 H0 V16" fill="none" stroke="rgba(100,116,139,.12)" />
          </pattern>
          <pattern id="balcony-lines" width="14" height="14" patternUnits="userSpaceOnUse">
            <path d="M0 14 L14 0" stroke="rgba(71,85,105,.18)" />
          </pattern>
          <linearGradient id="base-floor" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%" stopColor="#f8fafc" />
            <stop offset="52%" stopColor="#f1f5f9" />
            <stop offset="100%" stopColor="#e2e8f0" />
          </linearGradient>
          <linearGradient id="balcony-floor" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%" stopColor="#ecfeff" />
            <stop offset="100%" stopColor="#cffafe" />
          </linearGradient>
        </defs>

        <rect x="24" y="24" width="712" height="452" rx="22" fill="#ffffff" filter="url(#soft-shadow)" />
        <path d="M54 70 H596 V136 H716 V434 H54 Z" fill="url(#base-floor)" />
        <path d="M54 70 H596 V136 H716 V434 H54 Z" fill="url(#floor-grid)" opacity=".72" />
        <path d="M596 136 H716 V434 H596 Z" fill="url(#balcony-floor)" opacity=".82" />
        <path d="M596 136 H716 V434 H596 Z" fill="url(#balcony-lines)" opacity=".6" />

        {rooms.map((room) => {
          const Icon = roomIcons[room.name];
          const risk = roomRisks.find((item) => item.room === room.name)?.risk ?? 35;
          const active =
            hoveredRoom === room.name ||
            selectedRoom === room.name ||
            activeReading.room === room.name;

          return (
            <motion.g
              key={room.name}
              onMouseEnter={() => setHoveredRoom(room.name)}
              onMouseLeave={() => setHoveredRoom(null)}
              onClick={() => setSelectedRoom(room.name)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  setSelectedRoom(room.name);
                }
              }}
              tabIndex={0}
              role="button"
              aria-label={`ดูข้อมูลความเสี่ยง ${roomThai[room.name]}`}
              className="cursor-pointer outline-none"
              initial={false}
              animate={{ opacity: active ? 1 : 0.95 }}
            >
              <motion.path
                d={room.d}
                fill={room.name === "Balcony" ? "rgba(224,242,254,.48)" : "rgba(255,255,255,.26)"}
                stroke={active ? riskColor(risk) : "rgba(15,23,42,.62)"}
                strokeWidth={active ? 3 : 1.4}
              />
              <motion.path
                d={room.d}
                fill={riskColor(risk)}
                initial={false}
                animate={{ opacity: active ? roomHeatOpacity(risk) + 0.05 : roomHeatOpacity(risk) }}
              />
              <foreignObject x={room.label.x} y={room.label.y - 24} width="176" height="40">
                <div className="flex w-fit items-center gap-2 rounded-md bg-white/88 px-3 py-1.5 text-[13px] font-extrabold text-slate-950 shadow-sm ring-1 ring-slate-200">
                  <Icon className="h-4 w-4 text-slate-700" />
                  <span>{roomThai[room.name]}</span>
                </div>
              </foreignObject>
              {detailed && (
                <text x={room.label.x} y={room.label.y + 26} className="fill-slate-800 text-[11px] font-bold">
                  เสี่ยง {Math.round(risk)}% / ใช้งาน {roomRisks.find((item) => item.room === room.name)?.activity ?? 30}%
                </text>
              )}
            </motion.g>
          );
        })}

        <HeatmapOverlay points={visibleHeatPoints} />
        <ArchitecturalDetails />
        <FootstepTrail path={sampleMovementPath} compact={compact} />

        {nearFalls.map((reading) => (
          <NearFallMarker key={reading.timestamp} reading={reading} />
        ))}

        {rooms.map((room) => (
          <path
            key={`click-${room.name}`}
            d={room.d}
            fill="rgba(255,255,255,0.001)"
            stroke="none"
            role="button"
            tabIndex={0}
            data-room-click-target={room.name}
            aria-label={`ดูข้อมูลความเสี่ยง ${roomThai[room.name]}`}
            className="cursor-pointer"
            onClick={() => setSelectedRoom(room.name)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setSelectedRoom(room.name);
              }
            }}
          />
        ))}

        <RiskLegend />
      </svg>

      {hoveredRoom && (
        <div className="pointer-events-none absolute left-5 top-5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-950 shadow-lg">
          <span className="font-extrabold">{roomThai[hoveredRoom]}</span>
          <span className="ml-2 font-semibold text-slate-700">
            เสี่ยง {roomRisks.find((room) => room.room === hoveredRoom)?.risk}%
          </span>
        </div>
      )}

      {selectedRoom && (
        <RoomGuidancePanel
          room={selectedRoom}
          risk={roomRisks.find((room) => room.room === selectedRoom)?.risk ?? 35}
          onClose={() => setSelectedRoom(null)}
        />
      )}
    </div>
  );
}

function roomHeatOpacity(risk: number) {
  if (risk > 70) return 0.2;
  if (risk >= 36) return 0.12;
  return 0.05;
}

function ArchitecturalDetails() {
  return (
    <g>
      <path
        d="M54 70 H596 V136 H716 V434 H54 Z"
        fill="none"
        stroke="#1f2937"
        strokeLinejoin="round"
        strokeWidth="5"
      />
      <path
        d="M282 70 V226 H246 V286 H54 M206 286 V434 M366 226 V434 M366 268 H596 M596 136 V434 M596 238 H716 M596 334 H716"
        fill="none"
        stroke="#334155"
        strokeLinejoin="round"
        strokeWidth="4"
      />

      <path d="M246 226 A54 54 0 0 1 300 280" fill="none" stroke="#475569" strokeWidth="2" />
      <path d="M206 334 A48 48 0 0 0 254 286" fill="none" stroke="#475569" strokeWidth="2" />
      <path d="M366 340 A54 54 0 0 1 312 286" fill="none" stroke="#475569" strokeWidth="2" />
      <path d="M596 286 A48 48 0 0 0 548 238" fill="none" stroke="#475569" strokeWidth="2" />

      <rect x="86" y="116" width="132" height="78" rx="14" fill="none" stroke="#64748b" strokeOpacity=".75" strokeWidth="1.4" />
      <rect x="104" y="130" width="92" height="30" rx="7" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <rect x="224" y="104" width="42" height="130" rx="8" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <rect x="92" y="330" width="82" height="38" rx="8" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <rect x="82" y="382" width="78" height="44" rx="22" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <circle cx="182" cy="402" r="20" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <rect x="404" y="318" width="154" height="34" rx="7" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <rect x="538" y="366" width="34" height="54" rx="8" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <circle cx="434" cy="338" r="12" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <rect x="418" y="136" width="120" height="74" rx="16" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <rect x="542" y="154" width="40" height="74" rx="12" fill="none" stroke="#64748b" strokeOpacity=".65" strokeWidth="1.4" />
      <rect x="300" y="246" width="38" height="92" rx="8" fill="none" stroke="#64748b" strokeOpacity=".48" strokeWidth="1.4" />
      <rect x="238" y="358" width="86" height="36" rx="9" fill="none" stroke="#64748b" strokeOpacity=".48" strokeWidth="1.4" />
      <foreignObject x="648" y="378" width="46" height="34">
        <div className="text-slate-700"><Waves className="h-6 w-6" /></div>
      </foreignObject>
    </g>
  );
}

function RiskLegend() {
  return (
    <foreignObject x="536" y="10" width="184" height="58">
      <div className="rounded-lg border border-slate-300 bg-white/95 p-2 text-[10px] text-slate-800 shadow-sm backdrop-blur">
        <div className="mb-1 font-extrabold text-slate-950">แผนที่ความเสี่ยง</div>
        <div className="grid grid-cols-3 gap-1">
          <span className="rounded bg-lime-400 px-1.5 py-0.5 font-bold text-slate-950">ปกติ</span>
          <span className="rounded bg-yellow-300 px-1.5 py-0.5 font-bold text-slate-950">กลาง</span>
          <span className="rounded bg-red-500 px-1.5 py-0.5 font-bold text-white">สูง</span>
        </div>
      </div>
    </foreignObject>
  );
}

function RoomGuidancePanel({
  room,
  risk,
  onClose,
}: {
  room: RoomName;
  risk: number;
  onClose: () => void;
}) {
  const guidance = roomGuidance[room];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.22 }}
      className="absolute bottom-5 left-5 right-5 z-20 rounded-2xl border border-slate-200 bg-white/96 p-4 text-slate-950 shadow-[0_18px_45px_rgba(15,23,42,0.18)] backdrop-blur"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-extrabold text-cyan-700">กดบนแผนที่เพื่อดูรายละเอียด</div>
          <h3 className="mt-1 text-lg font-extrabold">{roomThai[room]}</h3>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-extrabold text-slate-700 hover:bg-slate-100"
        >
          ปิด
        </button>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-[1fr_.9fr]">
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
          <div className="text-xs font-extrabold text-amber-800">สิ่งที่ควรระวัง</div>
          <p className="mt-1 text-sm font-bold leading-6 text-slate-950">{guidance.focus}</p>
          <ul className="mt-2 space-y-1 text-sm font-semibold text-slate-700">
            {guidance.cautions.map((item) => (
              <li key={item}>- {item}</li>
            ))}
          </ul>
        </div>

        <div className="rounded-xl border border-cyan-200 bg-cyan-50 p-3">
          <div className="text-xs font-extrabold text-cyan-800">ลักษณะการเดิน</div>
          <p className="mt-1 text-sm font-bold leading-6 text-slate-950">{guidance.walking}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {guidance.activities.map((activity) => (
              <span
                key={activity.code}
                className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-extrabold text-slate-800"
              >
                {activity.code} · {activity.label}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-3 flex items-center gap-3 text-xs font-bold text-slate-700">
        <span>ความเสี่ยงห้องนี้</span>
        <div className="h-2 flex-1 rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-amber-300 to-rose-500"
            style={{ width: `${risk}%` }}
          />
        </div>
        <span className="text-slate-950">{risk}%</span>
      </div>
    </motion.div>
  );
}
