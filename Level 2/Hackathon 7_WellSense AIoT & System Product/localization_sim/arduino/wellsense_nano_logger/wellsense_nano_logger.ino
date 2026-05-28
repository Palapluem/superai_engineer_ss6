#include <Arduino_Modulino.h>

ModulinoMovement movement;
ModulinoDistance distance;

const unsigned long SAMPLE_PERIOD_MS = 100;
const float G_TO_MPS2 = 9.80665f;
const float DEG_TO_RAD_S = 0.017453292519943295f;
const int MAX_DISTANCE_MM = 1200;

unsigned long lastSampleMs = 0;

void setup() {
  Serial.begin(115200);
  unsigned long serialStart = millis();
  while (!Serial && millis() - serialStart < 3000) {
    delay(10);
  }

  Modulino.begin();
  bool imuOk = movement.begin();
  bool distanceOk = distance.begin();

  Serial.println("# WellSense Arduino Nano + Modulino packet logger");
  Serial.print("# movement_ok=");
  Serial.println(imuOk ? 1 : 0);
  Serial.print("# distance_ok=");
  Serial.println(distanceOk ? 1 : 0);
  Serial.println("time_ms,ax,ay,az,gx,gy,gz,distance_mm,distance_valid");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_PERIOD_MS) {
    return;
  }
  lastSampleMs = now;

  bool imuValid = movement.available();
  if (imuValid) {
    movement.update();
  }

  bool distanceValid = distance.available();
  int distanceMm = distanceValid ? (int)round(distance.get()) : MAX_DISTANCE_MM;

  float ax = 0.0f;
  float ay = 0.0f;
  float az = G_TO_MPS2;
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;

  if (imuValid) {
    ax = movement.getX() * G_TO_MPS2;
    ay = movement.getY() * G_TO_MPS2;
    az = movement.getZ() * G_TO_MPS2;
    gx = movement.getRoll() * DEG_TO_RAD_S;
    gy = movement.getPitch() * DEG_TO_RAD_S;
    gz = movement.getYaw() * DEG_TO_RAD_S;
  }

  Serial.print(now);
  Serial.print(",");
  Serial.print(ax, 5);
  Serial.print(",");
  Serial.print(ay, 5);
  Serial.print(",");
  Serial.print(az, 5);
  Serial.print(",");
  Serial.print(gx, 5);
  Serial.print(",");
  Serial.print(gy, 5);
  Serial.print(",");
  Serial.print(gz, 5);
  Serial.print(",");
  Serial.print(distanceMm);
  Serial.print(",");
  Serial.println(distanceValid ? 1 : 0);
}
