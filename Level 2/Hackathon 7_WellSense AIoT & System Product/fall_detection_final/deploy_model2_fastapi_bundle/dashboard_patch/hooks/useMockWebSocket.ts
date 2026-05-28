"use client";

import { useEffect } from "react";
import { generateReading } from "@/lib/mock-data";
import type { SensorReading } from "@/lib/types";
import { useMonitoringStore } from "@/store/monitoring-store";

const MODEL2_API_URL = process.env.NEXT_PUBLIC_MODEL2_API_URL ?? "http://localhost:8000";

async function fetchLiveReading(): Promise<SensorReading | null> {
  try {
    const response = await fetch(`${MODEL2_API_URL}/dashboard/latest`, {
      cache: "no-store",
    });
    if (!response.ok) return null;
    const reading = (await response.json()) as SensorReading;
    if (typeof reading.fall_risk !== "number") return null;
    return {
      ...reading,
      timestamp: reading.timestamp || new Date().toISOString(),
    };
  } catch {
    return null;
  }
}

export function useMockWebSocket() {
  const applyReading = useMonitoringStore((state) => state.applyReading);

  useEffect(() => {
    let active = true;

    async function tick() {
      const currentIndex = useMonitoringStore.getState().streamIndex;
      const liveReading = await fetchLiveReading();
      if (!active) return;
      applyReading(liveReading ?? generateReading(currentIndex, new Date()));
    }

    void tick();
    const timer = window.setInterval(() => {
      void tick();
    }, 3200);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [applyReading]);
}
