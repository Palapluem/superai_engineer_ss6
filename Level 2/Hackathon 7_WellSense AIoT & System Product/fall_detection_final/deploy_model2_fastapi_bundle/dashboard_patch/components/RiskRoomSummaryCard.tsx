"use client";

import { motion } from "framer-motion";
import { ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RoomRisk } from "@/lib/types";
import { cn } from "@/lib/utils";

interface RiskRoomSummaryCardProps {
  riskScore: number;
  rooms: RoomRisk[];
  roomLabel: (room: string) => string;
}

export function RiskRoomSummaryCard({
  riskScore,
  rooms,
  roomLabel,
}: RiskRoomSummaryCardProps) {
  const tone = riskTone(riskScore);

  return (
    <Card className={cn("overflow-hidden", tone.card)}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className={cn("rounded-xl p-2.5", tone.iconBg)}>
              <ShieldAlert className={cn("h-5 w-5", tone.icon)} />
            </div>
            <div>
              <CardTitle>ความเสี่ยงล้ม</CardTitle>
              <p className="mt-1 text-sm font-semibold text-slate-700">ประเมินล่าสุด + ห้องเสี่ยงที่สุด</p>
            </div>
          </div>
          <Badge variant={tone.badge}>{tone.label}</Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div>
          <div className="flex items-end justify-between gap-3">
            <div className="text-4xl font-extrabold tracking-normal text-slate-950">{riskScore}</div>
            <div className="pb-1 text-lg font-extrabold text-slate-950">{riskScore}%</div>
          </div>
          <div className="mt-3 h-2.5 rounded-full bg-slate-100" role="progressbar" aria-label="ความเสี่ยงล้ม" aria-valuemin={0} aria-valuemax={100} aria-valuenow={riskScore}>
            <motion.div
              className={cn("h-full rounded-full", tone.bar)}
              initial={{ width: 0 }}
              animate={{ width: `${riskScore}%` }}
              transition={{ type: "spring", stiffness: 80, damping: 18 }}
            />
          </div>
        </div>

        <div className="space-y-3 border-t border-slate-200 pt-4">
          <div className="text-sm font-extrabold text-slate-950">ห้องที่เสี่ยงที่สุด</div>
          {rooms.map((room, index) => (
            <div key={room.room} className="rounded-lg border border-slate-200 bg-white p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Badge variant={index === 0 ? "danger" : "soft"}>#{index + 1}</Badge>
                  <span className="font-bold text-slate-950">{roomLabel(room.room)}</span>
                </div>
                <span className="text-sm font-extrabold text-slate-800">เสี่ยง {room.risk}%</span>
              </div>
              <div className="mt-3 h-2 rounded-full bg-slate-900">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-amber-300 to-rose-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${room.risk}%` }}
                  transition={{ type: "spring", stiffness: 80, damping: 18, delay: index * 0.04 }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function riskTone(riskScore: number) {
  if (riskScore > 70) {
    return {
      label: "สูง",
      badge: "danger" as const,
      card: "border-rose-200 bg-rose-50/70",
      iconBg: "bg-rose-100",
      icon: "text-rose-700",
      bar: "bg-rose-500",
    };
  }

  if (riskScore >= 36) {
    return {
      label: "ปานกลาง",
      badge: "warning" as const,
      card: "border-amber-200 bg-amber-50/70",
      iconBg: "bg-amber-100",
      icon: "text-amber-700",
      bar: "bg-amber-400",
    };
  }

  return {
    label: "ต่ำ",
    badge: "safe" as const,
    card: "border-emerald-200 bg-emerald-50/70",
    iconBg: "bg-emerald-100",
    icon: "text-emerald-700",
    bar: "bg-emerald-500",
  };
}
