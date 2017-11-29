#include <Ethernet.h>
#include <Wire.h>

#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"

#define SERVER {192, 168, 10, 100}
#define SERVER_PORT 6006
#define INTERRUPT_PIN 2
#define LED_PIN 13

#define ID             0
#define X_GYRO_OFFSET  83
#define Y_GYRO_OFFSET  5
#define Z_GYRO_OFFSET  32
#define Z_ACCEL_OFFSET 2102
#define MAC {0x90, 0xA2, 0xDA, 0x10, 0x26, 0x82 + ID}
#define IP  {192, 168, 10, 87 + ID}

EthernetClient client;
MPU6050 mpu;

byte mac[] = MAC;
byte ip[] = IP;
byte packet[] = {'$', ID, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, '\r', '\n'};
byte header[] = {'$', ID, sizeof(packet), '\r', '\n'};
uint8_t packetCount = 0;
uint16_t pos;

bool blinkState = false;
uint8_t mpuIntStatus;
uint8_t devStatus;
uint16_t packetSize;
uint16_t fifoCount;
uint8_t fifoBuffer[64];

Quaternion q;
VectorInt16 aa;
VectorInt16 aaReal;
VectorInt16 aaWorld;
VectorFloat gravity;

volatile bool mpuInterrupt = false;
void dmpDataReady() {
    mpuInterrupt = true;
}

void write(byte* buf, uint16_t len) {
    if (client.connected()) {
        client.write(buf, len);
    }
}

void setup() {
    Wire.begin();
    Wire.setClock(400000);

    Serial.begin(115200);

    Ethernet.begin(mac, ip);
    client.connect(SERVER, SERVER_PORT);
    write(header, sizeof(header));
    
    pinMode(INTERRUPT_PIN, INPUT);
    pinMode(LED_PIN, OUTPUT);

    mpu.initialize();
    mpu.dmpInitialize();

    mpu.setXGyroOffset(X_GYRO_OFFSET);
    mpu.setYGyroOffset(Y_GYRO_OFFSET);
    mpu.setZGyroOffset(Z_GYRO_OFFSET);
    mpu.setZAccelOffset(Z_ACCEL_OFFSET);

    mpu.setDMPEnabled(true);

    mpuIntStatus = mpu.getIntStatus();
    packetSize = mpu.dmpGetFIFOPacketSize();
    attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
}

void loop() {
    while (!mpuInterrupt && fifoCount < packetSize) {}

    mpuInterrupt = false;
    mpuIntStatus = mpu.getIntStatus();
    fifoCount = mpu.getFIFOCount();
    
    if ((mpuIntStatus & 0x10) || fifoCount == 1024) {
        mpu.resetFIFO();
        Serial.println(F("FIFO overflow!"));

    } else if (mpuIntStatus & 0x02) {
        while (fifoCount < packetSize) {
            fifoCount = mpu.getFIFOCount();
        }

        mpu.getFIFOBytes(fifoBuffer, packetSize);
        fifoCount -= packetSize;

        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetAccel(&aa, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
        mpu.dmpGetLinearAccelInWorld(&aaWorld, &aaReal, &q);

        pos = 2;
        packet[pos++] = packetCount++;
        packet[pos++] = (byte)(aaWorld.x & 0xFF);
        packet[pos++] = (byte)(aaWorld.x >> 8) & 0xFF;
        packet[pos++] = (byte)(aaWorld.y & 0xFF);
        packet[pos++] = (byte)(aaWorld.y >> 8) & 0xFF;
        packet[pos++] = (byte)(aaWorld.z & 0xFF);
        packet[pos++] = (byte)(aaWorld.z >> 8) & 0xFF;

        write(packet, sizeof(packet));

        blinkState = !blinkState;
        digitalWrite(LED_PIN, blinkState);
    }
}
