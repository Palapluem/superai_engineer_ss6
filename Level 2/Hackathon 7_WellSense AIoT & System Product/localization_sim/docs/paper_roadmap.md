# Paper roadmap for DulAe / WellSense

This note lists papers that are useful beyond the current static-vs-transition
activity decision. Use it to justify the project design, choose datasets, and
explain future work.

## Recommended reading order

1. SisFall dataset
2. MobiAct or UP-Fall dataset
3. Transition-aware HAR / postural transition detection
4. Wearable IMU fall-risk and gait papers
5. Indoor localization / map-matched pedestrian dead reckoning
6. Arduino Nano 33 BLE Sense / edge TinyML papers

## 1. Dataset baseline and data format

### SisFall: A Fall and Movement Dataset

Link: https://www.mdpi.com/1424-8220/17/1/198

Why it matters:

- Closest to our current dataset formatting.
- Uses ADL IDs (`Dxx`) and fall IDs (`Fxx`).
- Includes both young adults and elderly participants.
- Good support for our `dataset_formatter.py` output.

How to use in the project:

- Reference it as the main benchmark dataset.
- Keep our converted sensor rows compatible with its 6-axis/9-axis numeric
  row style.
- Use their activity naming as baseline, but simplify our prototype labels to
  `static`, `transition`, `walking/gait`, and `fall-like transition`.

### MobiAct Dataset

Link: https://www.scitepress.org/papers/2016/57924/57924.pdf

Why it matters:

- Popular benchmark for smartphone/wearable activity recognition and fall
  detection.
- Includes ADLs and fall-like movements with accelerometer and gyroscope.
- Useful if we want a larger dataset than SisFall for model comparison.

How to use:

- Good for model benchmarking and feature comparison.
- Useful source for ADL vs fall protocol design.

### UP-Fall Detection Dataset

Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC6539235/

Why it matters:

- Multimodal dataset: wearable sensors, ambient sensors, vision.
- Useful for explaining why our first prototype intentionally avoids camera
  privacy issues while still following fall-detection benchmark practice.
- Paper explicitly discusses public fall-detection dataset limitations.

How to use:

- Cite for fall-detection dataset landscape.
- Use as future-work reference if the team later adds ambient sensors or
  camera-free room sensors.

### UMAFall Dataset

Link: https://www.sciencedirect.com/science/article/pii/S1877050917312899

Why it matters:

- Multisensor fall/ADL dataset with body-worn devices at several locations.
- Useful for sensor placement discussion.

How to use:

- Cite when explaining why waist/chest/ankle/wrist placement changes model
  behavior.

## 2. Static posture vs transition detection

### A multi-resolution investigation for postural transition detection and quantification using a single wearable

Link: https://pubmed.ncbi.nlm.nih.gov/27513738/

Why it matters:

- Supports our new consult direction: detect transitions such as sit-to-stand
  and stand-to-sit rather than forcing static posture classes.
- Uses a single lower-back accelerometer, close to a practical wearable setup.

How to use:

- Justifies `static -> transition -> static` as a real event.
- Good support for state-machine logic.

### A Hybrid Deep Residual Network for Efficient Transitional Activity Recognition Based on Wearable Sensors

Link: https://www.mdpi.com/2076-3417/12/10/4988

Why it matters:

- Explicitly treats transitional activities as their own class.
- Useful for future ML model after rule-based prototype.

How to use:

- Supports future model classes:

```text
static
walking/gait
posture_transition
fall_like_transition
```

### HMM Adaptation for Improving a Human Activity Recognition System

Link: https://www.mdpi.com/1999-4893/9/3/60

Why it matters:

- Supports using temporal state models instead of independent frame-by-frame
  classification.
- Helpful for the "cannot sit again unless a transition happened first" idea.

How to use:

- Cite when explaining state constraints:

```text
static -> static = merge
static -> transition -> static = new event
```

## 3. Gait and fall-risk assessment

### Wearable inertial sensors to measure gait and posture characteristic differences in older adult fallers and non-fallers

Link: https://www.sciencedirect.com/science/article/pii/S0966636219301663

Why it matters:

- Review paper focused on older adult fallers vs non-fallers.
- Reports that wearable inertial measures correlate with fall-risk tests.
- Notes lower back/trunk sensor positions are often useful.

How to use:

- Main support for our gait-risk module and heatmap idea.
- Supports using waist/lower-back placement for gait stability features.

### Wearable Sensor-Based Prediction Model of Timed up and Go Test in Older Adults

Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC8540088/

Why it matters:

- Uses IMU data from older adults to estimate TUG-related fall-risk score.
- Recommends a single pelvis IMU as a practical placement tradeoff.

How to use:

- Supports our L4-L5 / waist placement recommendation.
- Useful for future "mobility score" dashboard metric.

### Sensor-Based Assessment of Falls Risk of the Timed Up and Go in Real-World Settings

Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC6840247/

Why it matters:

- Large real-world QTUG dataset across clinical and community settings.
- Shows sensor-based measures can quantify falls risk, frailty, and mobility
  impairment.

How to use:

- Strong clinical/product justification for a dashboard fall-risk score.

### Timed Up and Go and Six-Minute Walking Tests with Wearable Inertial Sensor

Link: https://www.mdpi.com/1424-8220/20/11/3207

Why it matters:

- Older nursing-home residents with wearable IMU.
- Combines TUG, six-minute walk, gait variability, and future fall prediction.

How to use:

- Supports our "walking pattern anomaly" and longitudinal dashboard direction.

## 4. Indoor x,y localization and heatmap

### A Novel Map-Based Dead-Reckoning Algorithm for Indoor Localization

Link: https://www.mdpi.com/2224-2708/3/1/44

Why it matters:

- Uses step-counting dead reckoning with map matching and particle filtering.
- Very relevant to our current localization simulator.

How to use:

- Cite for the next version after our rule-based estimator:

```text
IMU dead reckoning + known map + particle filter = better x,y
```

### Indoor Positioning System Based on Chest-Mounted IMU

Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC6359165/

Why it matters:

- Chest-mounted IMU with map matching for indoor positioning.
- Good support for map-constrained wearable localization.

How to use:

- Cite when explaining why a known floor plan can reduce localization drift.

### Applied Indoor Localization: Map-based, Infrastructure-free, with Divergence Mitigation and Smoothing

Link: https://www.ri.cmu.edu/publications/applied-indoor-localization-map-based-infrastructure-free-with-divergence-mitigation-and-smoothing/

Why it matters:

- Low-cost body-mounted IMU, steps, heading change, map registration, and
  particle filter.
- Good future-work reference for infrastructure-free localization.

How to use:

- Supports dashboard coordinate heatmap without BLE/UWB anchors.

## 5. Arduino Nano 33 BLE Sense / edge AI

### Human motion activity recognition and pattern analysis using compressed deep neural networks

Link: https://www.tandfonline.com/doi/full/10.1080/21681163.2024.2331052

Why it matters:

- Uses Arduino Nano 33 BLE Sense IMU for activity recognition.
- Supports our hardware choice for wearable motion classification.

How to use:

- Cite for on-device / TinyML future direction.

### Design of a Wearable Healthcare Emergency Detection Device for Elder Persons

Link: https://www.mdpi.com/2076-3417/12/5/2345

Why it matters:

- Uses Arduino Nano 33 BLE Sense as wearable elderly emergency/fall detection
  device.
- Very close to our hardware story.

How to use:

- Cite in the system design section.

## Recommended changes to the current project overview

Based on consult feedback, change the activity taxonomy in the poster.

Current:

```text
Daily Activity: เดินปกติ, เดินขึ้นลงบันได, ยืน/นั่งอ่าน, สะดุด, ขึ้นรถ
Fall Class: 5 fall types
```

Recommended:

```text
Mobility State:
1. Static / no immediate risk
   - standing, sitting, lying still are merged
2. Walking / gait
   - normal, slow, limping, unstable walking
3. Posture transition
   - sit-to-stand, stand-to-sit, get-up, lie-to-sit
4. Fall-like transition
   - collapse, rapid orientation change, high jerk
5. Recovery / stillness after event
   - stillness after transition, possible alert window
```

The fall-risk logic becomes:

```text
static alone -> no alert
walking with unstable gait -> risk score
posture transition -> monitor
fall-like transition + stillness -> alert candidate
repeated static label -> merge, not a new event
```

## What to implement next

1. Add a `mobility_state` field after preprocessing:

```text
static
walking
posture_transition
fall_like_transition
recovery_stillness
```

2. Add a simple state machine:

```text
static -> static: merge
static -> transition: event_start
transition -> static: event_end
transition -> fall_like: alert_candidate
fall_like -> stillness: alert_candidate_high
```

3. Keep `Dxx/Fxx` dataset compatibility, but train/evaluate with coarser
classes first:

```text
D05/D06/static-like ADL -> static
D01-D04 walking-like ADL -> walking/gait
D07 -> posture_transition
F01-F02 -> fall_like_transition
```

4. Dashboard should show:

```text
current mobility state
transition events
fall-like alert candidates
gait risk score
location heatmap
```
