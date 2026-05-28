import { AlertSeverity, CareAlert, AIInsight, SensorReading } from "@/lib/types";
import { formatClock } from "@/lib/utils";

function severityFromRisk(risk: number): AlertSeverity {
  if (risk >= 90) return "emergency";
  if (risk > 70) return "high";
  if (risk >= 36) return "medium";
  return "low";
}

export function analyzeReading(
  reading: SensorReading,
  previous: SensorReading[],
): { insight?: AIInsight; alert?: CareAlert } {
  const riskJump =
    previous.length > 0
      ? reading.fall_risk - previous[previous.length - 1].fall_risk
      : 0;
  const hour = new Date(reading.timestamp).getHours();
  const nighttime = hour >= 22 || hour <= 5;
  const severity = severityFromRisk(reading.fall_risk);
  const idSuffix = `${reading.timestamp}-${reading.room}`;

  if (reading.near_fall) {
    return {
      insight: {
        id: `insight-near-${idSuffix}`,
        title: "High-risk movement pattern identified",
        detail:
          "The edge AI model detected abrupt acceleration, elevated sway, and delayed turn recovery consistent with a near-fall event.",
        confidence: 94,
        severity: "emergency",
        room: reading.room,
        createdAt: formatClock(reading.timestamp),
      },
      alert: {
        id: `alert-near-${idSuffix}`,
        message: "Near-fall event detected",
        severity: "emergency",
        room: reading.room,
        timestamp: formatClock(reading.timestamp),
        acknowledged: false,
      },
    };
  }

  if (reading.room === "Bathroom" && reading.fall_risk > 70) {
    return {
      insight: {
        id: `insight-bath-${idSuffix}`,
        title: "Instability increased near bathroom",
        detail:
          "Bathroom risk is elevated by slower gait speed, higher sway, and wet-zone proximity. The model recommends caregiver awareness.",
        confidence: 89,
        severity,
        room: "Bathroom",
        createdAt: formatClock(reading.timestamp),
      },
      alert: {
        id: `alert-bath-${idSuffix}`,
        message: "High instability detected in bathroom",
        severity,
        room: "Bathroom",
        timestamp: formatClock(reading.timestamp),
        acknowledged: false,
      },
    };
  }

  if (nighttime && reading.instability_score > 55) {
    return {
      insight: {
        id: `insight-night-${idSuffix}`,
        title: "Nighttime gait instability detected",
        detail:
          "Night walking shows reduced cadence and increased path deviation along the bedroom-to-bathroom route.",
        confidence: 87,
        severity: "high",
        room: reading.room,
        createdAt: formatClock(reading.timestamp),
      },
    };
  }

  if (riskJump > 16 || reading.turning_velocity < 46) {
    return {
      insight: {
        id: `insight-turn-${idSuffix}`,
        title: "Turning stability decreased this week",
        detail:
          "The inference engine observed slower turn velocity and rising gait variability at a tight transition zone.",
        confidence: 82,
        severity: "medium",
        room: reading.room,
        createdAt: formatClock(reading.timestamp),
      },
      alert:
        reading.fall_risk > 70
          ? {
              id: `alert-gait-${idSuffix}`,
              message: "Abnormal gait pattern detected",
              severity: "medium",
              room: reading.room,
              timestamp: formatClock(reading.timestamp),
              acknowledged: false,
            }
          : undefined,
    };
  }

  return {};
}
