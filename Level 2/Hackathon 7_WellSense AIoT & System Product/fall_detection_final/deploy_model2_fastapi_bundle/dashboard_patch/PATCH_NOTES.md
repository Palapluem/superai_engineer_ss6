# Dashboard Patch Notes

Copy these files into the existing `fall-detect-dashboard` project with the same relative paths.

## Files

```text
hooks/useMockWebSocket.ts
lib/utils.ts
lib/ai-engine.ts
app/page.tsx
components/RiskRoomSummaryCard.tsx
components/RiskScoreCard.tsx
components/floorplan/CondoFloorplanMap.tsx
```

## Behavior

- The dashboard polls `http://localhost:8000/dashboard/latest`.
- If the Model 2 API is unavailable, the dashboard uses existing mock data.
- Dashboard risk thresholds match Model 2:

```text
low    : 0 - 35
medium : 36 - 70
high   : > 70
```

## Optional Environment Variable

Use this if the API is not hosted on localhost:

```text
NEXT_PUBLIC_MODEL2_API_URL=http://<api-host>:8000
```
