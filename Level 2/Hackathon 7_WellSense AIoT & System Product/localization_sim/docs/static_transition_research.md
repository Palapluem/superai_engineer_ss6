# Static vs transition activity research notes

Consult decision:

```text
static activity = standing / lying / sitting still -> no immediate fall risk
transition activity = posture change / instability / collapse-like motion -> fall risk candidate
```

## Short answer

For our current wearable IMU setup, it is safer to group standing, sitting, and
lying as `static/no_risk` unless the sensor mounting position is tightly
controlled.

Static postures can be detected in papers, but the result depends heavily on
sensor placement:

- thigh-mounted sensor: sitting vs standing is easier because thigh angle
  changes a lot.
- chest/trunk/waist sensor: lying vs upright is often detectable, but sitting
  vs standing can look similar.
- loose clothing / unknown placement: static posture labels become less
  reliable.

This supports a product rule:

```text
static -> no alert
static -> transition -> evaluate fall risk
transition -> static -> update state
static -> same static repeated -> ignore / merge
```

## Papers and why they matter

| Topic | Paper | Useful point for WellSense |
| --- | --- | --- |
| Arduino Nano 33 BLE Sense HAR | [Human motion activity recognition and pattern analysis using compressed deep neural networks](https://www.tandfonline.com/doi/full/10.1080/21681163.2024.2331052) | Uses Arduino Nano 33 BLE Sense IMU at 119 Hz for on-device activity recognition. Good support for using the board as a TinyML wearable. |
| Arduino Nano 33 BLE Sense fall detection | [Design of a Wearable Healthcare Emergency Detection Device for Elder Persons](https://www.mdpi.com/2076-3417/12/5/2345) | Uses a single Arduino Nano 33 BLE Sense board and neural network for elderly fall detection. Good reference for edge fall alert architecture. |
| Postural transition detection | [A multi-resolution investigation for postural transition detection and quantification using a single wearable](https://pubmed.ncbi.nlm.nih.gov/27513738/) | Detects sit-to-stand and stand-to-sit with one lower-back accelerometer. Reported transition detection accuracy is strong enough to justify focusing on transitions. |
| State/history model | [HMM Adaptation for Improving a Human Activity Recognition System](https://www.mdpi.com/1999-4893/9/3/60) | Uses inertial signals and HMMs to segment activity sequences: walking, stairs, sitting, standing, lying. Supports using temporal state models instead of frame-by-frame labels. |
| Transition-aware HAR | [A Hybrid Deep Residual Network for Efficient Transitional Activity Recognition Based on Wearable Sensors](https://www.mdpi.com/2076-3417/12/10/4988) | Explicitly separates basic activities from transitional activities such as sit-to-stand and stand-to-sit. Good support for adding transition classes. |
| Static posture limitations | [SVM-based posture identification with a single waist-located triaxial accelerometer](https://www.sciencedirect.com/science/article/abs/pii/S0957417413005058) | Important caveat: some static postures can produce similar signals with one accelerometer, so transitions help identify the posture context. |
| Static posture by sensor placement | [Detection of static and dynamic activities using uniaxial accelerometers](https://pure.eur.nl/en/publications/detection-of-static-and-dynamic-activities-using-uniaxial-acceler/) | Static posture recognition is feasible when sensors are mounted at useful body locations such as thigh and sternum. |
| Single accelerometer posture/transition | [Detection of daily postures and walking modalities using a single chest-mounted tri-axial accelerometer](https://www.sciencedirect.com/science/article/pii/S1350453318300626) | Shows that one chest-mounted accelerometer can detect standing, sitting, lying, walking, and several transitions, but placement is controlled. |

## Arduino Nano 33 BLE Sense support

Arduino Nano 33 BLE Sense Rev2 has a 9-axis IMU: BMI270 for 3-axis
accelerometer + gyroscope and BMM150 for magnetometer. Arduino's datasheet
describes this IMU data as usable for raw movement parameters and machine
learning.

Reference: [Arduino Nano 33 BLE Sense Rev2 datasheet](https://docs.arduino.cc/resources/datasheets/ABX00069-datasheet.pdf)

## Recommended model design

Use coarse states:

```text
STATIC
MOVING
POSTURE_TRANSITION
FALL_LIKE_TRANSITION
RECOVERY_OR_STILL_AFTER_EVENT
```

Do not force the first prototype to classify:

```text
standing vs sitting vs lying
```

unless sensor mounting is fixed and validated.

## State-machine rule

Example state constraints:

```text
STATIC -> STATIC
STATIC -> POSTURE_TRANSITION
POSTURE_TRANSITION -> STATIC
POSTURE_TRANSITION -> FALL_LIKE_TRANSITION
FALL_LIKE_TRANSITION -> RECOVERY_OR_STILL_AFTER_EVENT
RECOVERY_OR_STILL_AFTER_EVENT -> STATIC
```

Invalid or low-value transitions:

```text
sitting -> sitting
lying -> lying
standing -> standing
```

For our product, merge those into:

```text
static -> static
```

The useful event is not "the user is sitting again"; it is:

```text
static -> transition -> static
```

or:

```text
moving/static -> abnormal transition -> stillness
```

## Features to detect transition/fall risk

Window length:

```text
1-3 s for transition detection
3-5 s for gait/risk summary
```

Useful IMU features:

```text
acc_norm = sqrt(ax^2 + ay^2 + az^2)
gyro_norm = sqrt(gx^2 + gy^2 + gz^2)
jerk = diff(acc_norm)
orientation_delta = change in gravity direction / pitch-roll
stillness_after_event = low gyro + low dynamic acceleration after spike
```

Fall-risk heuristic:

```text
high acceleration/jerk
+ high angular velocity
+ orientation change
+ stillness after event
= fall-like transition candidate
```

## Team-facing wording

We should not spend effort separating standing, sitting, and lying in the first
version because a single wearable IMU may not reliably distinguish all static
postures unless the mounting position is controlled. The safer and more useful
approach is to classify static as no-risk, then detect transition events and
fall-like transitions. We can enforce state history so repeated static labels
are merged, and a new sitting/lying/standing event only matters after the system
observes a transition.
