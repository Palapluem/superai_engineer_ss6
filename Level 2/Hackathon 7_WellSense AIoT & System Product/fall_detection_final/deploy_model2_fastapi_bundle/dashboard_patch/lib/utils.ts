import { ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export function riskColor(score: number) {
  if (score > 70) return "#ef4444";
  if (score >= 36) return "#facc15";
  return "#22c55e";
}

export function severityTone(severity: string) {
  switch (severity) {
    case "emergency":
      return "border-rose-300 bg-rose-50 text-rose-900";
    case "high":
      return "border-orange-300 bg-orange-50 text-orange-950";
    case "medium":
      return "border-amber-300 bg-amber-50 text-amber-950";
    default:
      return "border-emerald-300 bg-emerald-50 text-emerald-950";
  }
}

export function formatClock(timestamp: string) {
  const date = new Date(timestamp);
  const bangkokDate = new Date(date.getTime() + 7 * 60 * 60 * 1000);
  return `${String(bangkokDate.getUTCHours()).padStart(2, "0")}:${String(
    bangkokDate.getUTCMinutes(),
  ).padStart(2, "0")}`;
}
