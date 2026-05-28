"use client";

import { LucideIcon } from "lucide-react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const tones = {
  risk: "from-rose-100 to-rose-50 text-rose-700",
  safe: "from-emerald-100 to-emerald-50 text-emerald-700",
  warning: "from-amber-100 to-orange-50 text-amber-700",
  care: "from-cyan-100 to-sky-50 text-cyan-700",
};

export function RiskScoreCard({
  label,
  value,
  icon: Icon,
  detail,
  subDetail,
  tone = "care",
  max = 100,
}: {
  label: string;
  value: number;
  icon: LucideIcon;
  detail: string;
  subDetail?: string;
  tone?: keyof typeof tones;
  max?: number;
}) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  const barColor = progressColor(pct, tone);
  const progressLabel = max === 100 ? `${pct}%` : `${value}/${max}`;

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className={cn("rounded-lg bg-gradient-to-br p-2", tones[tone])}>
            <Icon className="h-5 w-5" />
          </div>
          <span className="text-xs font-semibold text-slate-700">{detail}</span>
        </div>
        <div className="mt-4 text-sm font-bold text-slate-700">{label}</div>
        {subDetail && (
          <div className="mt-1 rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-2 text-xs font-extrabold text-amber-950">
            {subDetail}
          </div>
        )}
        <div className="mt-1 flex items-end justify-between gap-3">
          <div className="text-3xl font-semibold tracking-normal text-slate-950">{value}</div>
          <div className="pb-1 text-xs font-semibold text-slate-700">{progressLabel}</div>
        </div>
        <div
          className="mt-4 h-2 rounded-full bg-slate-100"
          role="progressbar"
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={max}
          aria-valuenow={value}
        >
          <motion.div
            className={cn("h-full rounded-full", barColor)}
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ type: "spring", stiffness: 80, damping: 18 }}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function progressColor(pct: number, tone: keyof typeof tones) {
  if (tone === "safe") {
    if (pct >= 70) return "bg-emerald-500";
    if (pct >= 36) return "bg-amber-400";
    return "bg-rose-500";
  }

  if (pct > 70) return "bg-rose-500";
  if (pct >= 36) return "bg-amber-400";
  return "bg-emerald-500";
}
