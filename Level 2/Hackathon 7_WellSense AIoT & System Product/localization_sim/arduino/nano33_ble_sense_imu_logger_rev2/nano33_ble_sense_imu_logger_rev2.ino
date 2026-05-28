#include <Arduino_BMI270_BMM150.h>

const unsigned long SAMPLE_PERIOD_MS = 100;
const float G_TO_MPS2 = 9.80665f;
const float DEG_PER_SEC_TO_RAD_PER_SEC = 0.017453292519943295f;
const int DISTANCE_PLACEHOLDER_MM = 1200;

unsigned long lastSampleMs = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  if (!IMU.begin()) {
    Serial.println("# ERROR: Failed to initialize IMU");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("# Arduino Nano 33 BLE Sense Rev2 IMU logger");
  Serial.println("# Units: ax/ay/az=m/s^2, gx/gy/gz=rad/s");
  Serial.println("time_ms,ax,ay,az,gx,gy,gz,distance_mm,distance_valid");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_PERIOD_MS) {
    return;
  }
  lastSampleMs = now;

  float ax_g = 0.0f;
  float ay_g = 0.0f;
  float az_g = 0.0f;
  float gx_dps = 0.0f;
  float gy_dps = 0.0f;
  float gz_dps = 0.0f;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax_g, ay_g, az_g);
  }
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx_dps, gy_dps, gz_dps);
  }

  Serial.print(now);
  Serial.print(",");
  Serial.print(ax_g * G_TO_MPS2, 5);
  Serial.print(",");
  Serial.print(ay_g * G_TO_MPS2, 5);
  Serial.print(",");
  Serial.print(az_g * G_TO_MPS2, 5);
  Serial.print(",");
  Serial.print(gx_dps * DEG_PER_SEC_TO_RAD_PER_SEC, 5);
  Serial.print(",");
  Serial.print(gy_dps * DEG_PER_SEC_TO_RAD_PER_SEC, 5);
  Serial.print(",");
  Serial.print(gz_dps * DEG_PER_SEC_TO_RAD_PER_SEC, 5);
  Serial.print(",");
  Serial.print(DISTANCE_PLACEHOLDER_MM);
  Serial.print(",");
  Serial.println(0);
}
