#include <Arduino.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "sisfall_model.h"

// Set to false if your sensor libraries already provide g and deg/s values.
constexpr bool INPUT_IS_RAW_BITS = true;

// Keep true when replaying original SisFall 200 Hz rows. Set false if your loop
// already samples at SISFALL_SAMPLE_RATE_HZ / SISFALL_SAMPLE_STRIDE.
constexpr bool APPLY_TRAINING_STRIDE = true;

constexpr uint16_t WINDOW_SAMPLES = SISFALL_DEFAULT_WINDOW_SAMPLES;
constexpr size_t LINE_BUFFER_SIZE = 192;

struct SisFallAccumulator {
    uint32_t rowCount;
    float totals[SISFALL_SIGNAL_COUNT];
    float totalsSq[SISFALL_SIGNAL_COUNT];
    float totalsAbs[SISFALL_SIGNAL_COUNT];
    float minimums[SISFALL_SIGNAL_COUNT];
    float maximums[SISFALL_SIGNAL_COUNT];
    float maxAbsValues[SISFALL_SIGNAL_COUNT];
    float previous[SISFALL_SIGNAL_COUNT];
    bool hasPrevious[SISFALL_SIGNAL_COUNT];
    float diffAbsTotals[SISFALL_SIGNAL_COUNT];
    float maxAbsDiffs[SISFALL_SIGNAL_COUNT];

    void reset() {
        rowCount = 0;
        for (uint8_t i = 0; i < SISFALL_SIGNAL_COUNT; ++i) {
            totals[i] = 0.0f;
            totalsSq[i] = 0.0f;
            totalsAbs[i] = 0.0f;
            minimums[i] = FLT_MAX;
            maximums[i] = -FLT_MAX;
            maxAbsValues[i] = 0.0f;
            previous[i] = 0.0f;
            hasPrevious[i] = false;
            diffAbsTotals[i] = 0.0f;
            maxAbsDiffs[i] = 0.0f;
        }
    }

    void update(const float sensorValues[SISFALL_SENSOR_VALUE_COUNT]) {
        float signalValues[SISFALL_SIGNAL_COUNT];
        for (uint8_t i = 0; i < SISFALL_SENSOR_VALUE_COUNT; ++i) {
            signalValues[i] = sensorValues[i];
        }
        for (uint8_t sensor = 0; sensor < SISFALL_SENSOR_COUNT; ++sensor) {
            const uint8_t base = sensor * 3;
            const float x = sensorValues[base];
            const float y = sensorValues[base + 1];
            const float z = sensorValues[base + 2];
            signalValues[SISFALL_SENSOR_VALUE_COUNT + sensor] = sqrtf(x * x + y * y + z * z);
        }

        rowCount += 1;
        for (uint8_t i = 0; i < SISFALL_SIGNAL_COUNT; ++i) {
            const float value = signalValues[i];
            totals[i] += value;
            totalsSq[i] += value * value;
            const float absValue = fabsf(value);
            totalsAbs[i] += absValue;
            if (value < minimums[i]) {
                minimums[i] = value;
            }
            if (value > maximums[i]) {
                maximums[i] = value;
            }
            if (absValue > maxAbsValues[i]) {
                maxAbsValues[i] = absValue;
            }
            if (hasPrevious[i]) {
                const float diff = fabsf(value - previous[i]);
                diffAbsTotals[i] += diff;
                if (diff > maxAbsDiffs[i]) {
                    maxAbsDiffs[i] = diff;
                }
            }
            previous[i] = value;
            hasPrevious[i] = true;
        }
    }

    bool buildFeatures(float features[SISFALL_FEATURE_COUNT]) const {
        if (rowCount == 0) {
            return false;
        }

        const uint32_t diffCount = rowCount > 1 ? rowCount - 1 : 0;
        uint16_t featureIndex = 0;
        for (uint8_t i = 0; i < SISFALL_SIGNAL_COUNT; ++i) {
            const float mean = totals[i] / rowCount;
            const float variance = fmaxf(0.0f, (totalsSq[i] / rowCount) - (mean * mean));
            const float stddev = sqrtf(variance);
            const float meanAbsDiff = diffCount > 0 ? diffAbsTotals[i] / diffCount : 0.0f;

            features[featureIndex++] = mean;
            features[featureIndex++] = stddev;
            features[featureIndex++] = minimums[i];
            features[featureIndex++] = maximums[i];
            features[featureIndex++] = maximums[i] - minimums[i];
            features[featureIndex++] = sqrtf(totalsSq[i] / rowCount);
            features[featureIndex++] = totalsAbs[i] / rowCount;
            features[featureIndex++] = maxAbsValues[i];
            features[featureIndex++] = meanAbsDiff;
            features[featureIndex++] = maxAbsDiffs[i];
        }
        return featureIndex == SISFALL_FEATURE_COUNT;
    }
};

SisFallAccumulator accumulator;
uint32_t validInputRows = 0;
char lineBuffer[LINE_BUFFER_SIZE];
size_t lineLength = 0;

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = expf(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(value);
    return z / (1.0f + z);
}

float predictFallProbability(const float features[SISFALL_FEATURE_COUNT]) {
    float score = SISFALL_MODEL_BIAS;
    for (uint16_t i = 0; i < SISFALL_FEATURE_COUNT; ++i) {
        const float scaled = (features[i] - SISFALL_SCALER_MEANS[i]) / SISFALL_SCALER_SCALES[i];
        score += SISFALL_MODEL_WEIGHTS[i] * scaled;
    }
    return sigmoid(score);
}

bool parseSensorLine(char *line, float values[SISFALL_SENSOR_VALUE_COUNT]) {
    char *cursor = line;
    for (uint8_t i = 0; i < SISFALL_SENSOR_VALUE_COUNT; ++i) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0' || *cursor == '\r' || *cursor == '\n' || *cursor == ';') {
            return false;
        }

        char *end = cursor;
        values[i] = strtof(cursor, &end);
        if (end == cursor) {
            return false;
        }
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == ',' || *cursor == ';') {
            ++cursor;
        }
    }
    return true;
}

void resetWindow() {
    accumulator.reset();
    validInputRows = 0;
}

void classifyWindow() {
    float features[SISFALL_FEATURE_COUNT];
    if (!accumulator.buildFeatures(features)) {
        return;
    }

    const float probability = predictFallProbability(features);
    Serial.print("fall_probability=");
    Serial.print(probability, 6);
    Serial.print(",label=");
    Serial.println(probability >= SISFALL_DECISION_THRESHOLD ? "fall" : "adl");
}

void processSensorValues(float values[SISFALL_SENSOR_VALUE_COUNT]) {
    if (INPUT_IS_RAW_BITS) {
        for (uint8_t i = 0; i < SISFALL_SENSOR_VALUE_COUNT; ++i) {
            values[i] *= SISFALL_UNIT_SCALES[i];
        }
    }

    validInputRows += 1;
    if (APPLY_TRAINING_STRIDE && ((validInputRows - 1) % SISFALL_SAMPLE_STRIDE != 0)) {
        return;
    }

    accumulator.update(values);
    if (accumulator.rowCount >= WINDOW_SAMPLES) {
        classifyWindow();
        resetWindow();
    }
}

void processLine(char *line) {
    float values[SISFALL_SENSOR_VALUE_COUNT];
    if (!parseSensorLine(line, values)) {
        return;
    }
    processSensorValues(values);
}

void pollSerial() {
    while (Serial.available() > 0) {
        const char c = static_cast<char>(Serial.read());
        if (c == '\n' || c == '\r') {
            if (lineLength > 0) {
                lineBuffer[lineLength] = '\0';
                processLine(lineBuffer);
                lineLength = 0;
            }
            continue;
        }
        if (lineLength < LINE_BUFFER_SIZE - 1) {
            lineBuffer[lineLength++] = c;
        } else {
            lineLength = 0;
        }
    }
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {
    }
    resetWindow();
    Serial.println("uno_q_sisfall ready");
    Serial.println("Send sensor rows as CSV in model input order.");
}

void loop() {
    pollSerial();
}
