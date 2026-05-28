#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

const char DEVICE_NAME[] = "WellSenseNano";
const char SERVICE_UUID[] = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char IMU_CHAR_UUID[] = "19b10001-e8f2-537e-4f6c-d104768a1214";

const unsigned long SAMPLE_PERIOD_MS = 100;
const float G_TO_MPS2 = 9.80665f;
const float DEG_PER_SEC_TO_RAD_PER_SEC = 0.017453292519943295f;

BLEService imuService(SERVICE_UUID);
BLECharacteristic imuCsvCharacteristic(IMU_CHAR_UUID, BLERead | BLENotify, 128);

unsigned long lastSampleMs = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000) {
    delay(10);
  }

  if (!IMU.begin()) {
    Serial.println("# ERROR: Failed to initialize IMU");
    while (true) {
      delay(1000);
    }
  }

  if (!BLE.begin()) {
    Serial.println("# ERROR: Failed to start BLE");
    while (true) {
      delay(1000);
    }
  }

  BLE.setLocalName(DEVICE_NAME);
  BLE.setDeviceName(DEVICE_NAME);
  BLE.setAdvertisedService(imuService);
  imuService.addCharacteristic(imuCsvCharacteristic);
  BLE.addService(imuService);
  imuCsvCharacteristic.writeValue((const unsigned char *)"time_ms,ax,ay,az,gx,gy,gz", 27);
  BLE.advertise();

  Serial.println("# WellSense Nano BLE IMU Rev1");
  Serial.println("# Advertising as WellSenseNano");
  Serial.println("# CSV payload: time_ms,ax,ay,az,gx,gy,gz");
}

void loop() {
  BLEDevice central = BLE.central();
  if (!central) {
    return;
  }

  Serial.print("# connected: ");
  Serial.println(central.address());

  while (central.connected()) {
    unsigned long now = millis();
    if (now - lastSampleMs < SAMPLE_PERIOD_MS) {
      BLE.poll();
      continue;
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

    char line[128];
    snprintf(
      line,
      sizeof(line),
      "%lu,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f",
      now,
      ax_g * G_TO_MPS2,
      ay_g * G_TO_MPS2,
      az_g * G_TO_MPS2,
      gx_dps * DEG_PER_SEC_TO_RAD_PER_SEC,
      gy_dps * DEG_PER_SEC_TO_RAD_PER_SEC,
      gz_dps * DEG_PER_SEC_TO_RAD_PER_SEC
    );

    imuCsvCharacteristic.writeValue((const unsigned char *)line, strlen(line));
    Serial.println(line);
  }

  Serial.println("# disconnected");
}
