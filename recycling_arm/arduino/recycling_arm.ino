#include <Servo.h>

Servo servos[4];
const byte servoPins[4] = {3, 5, 6, 9}; // base, shoulder, elbow, gripper

const int HOME[4] = {90, 90, 90, 90};
const int PICK[4] = {90, 70, 110, 40};
const int PLASTIC_BIN[4] = {90, 70, 110, 40};
const int PAPER_BIN[4]   = {45, 70, 110, 40};
const int METAL_BIN[4]   = {135, 70, 110, 40};
const int OTHER_BIN[4]   = {90, 110, 70, 80};

void moveTo(const int angles[4]) {
  for (byte i = 0; i < 4; i++) {
    servos[i].write(constrain(angles[i], 0, 180));
  }
  delay(700);
}
void handleCommand(String command) {
  command.trim();
  command.toLowerCase();

  if (command == "home") {
    moveTo(HOME);
  } else if (command == "pick") {
    moveTo(PICK);
  } else if (command == "drop plastic") {
    moveTo(PLASTIC_BIN);
  } else if (command == "drop paper") {
    moveTo(PAPER_BIN);
  } else if (command == "drop metal") {
    moveTo(METAL_BIN);
  } else if (command == "drop other") {
    moveTo(OTHER_BIN);
  } else {
    Serial.println("ERR");
    return;
  }

  Serial.println("OK");
}

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(1000);

  for (byte i = 0; i < 4; i++) {
    servos[i].attach(servoPins[i]);
  }

  moveTo(HOME);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
  String command = Serial.readStringUntil('\n');
  handleCommand(command);

  }
}