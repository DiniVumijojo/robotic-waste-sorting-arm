import cv2
import mediapipe as mp
import math
import time


CAMERA_INDEX = 0
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
STABLE_SECONDS = 0.7
COMMAND_COOLDOWN_SECONDS = 1.0
ENABLE_SERIAL = False

GESTURE_COMMANDS = {
    "open palm": "OPEN",
    "fist": "FIST",
    "point": "POINT",
    "unknown": "STOP",
}

FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_gesture(hand):
    """Return 'open palm', 'fist', 'point', or 'unknown'."""
    landmarks = hand.landmark
    extended = [
        distance(landmarks[tip_id], landmarks[0])
        > distance(landmarks[pip_id], landmarks[0])
        for tip_id, pip_id in zip(FINGER_TIPS, FINGER_PIPS)
    ]

    if all(extended):
        return "open palm"
    if not any(extended):
        return "fist"
    index, middle, ring, pinky = extended
    if index and not middle and not ring and not pinky:
        return "point"
    return "unknown"


def show_serial_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for port in ports:
        print(f"{port.device}: {port.description}")


def send_command(connection, command):
    """Send one newline-terminated command to the ESP32."""
    if connection is None:
        print(f"[SERIAL DISABLED] Would send: {command}")
        return
    connection.write((command + "\n").encode("utf-8"))
    connection.flush()
    print(f"Sent command: {command}")


def main():
    connection = None
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}. Try CAMERA_INDEX=1."
        )

    try:
        if ENABLE_SERIAL:
            try:
                connection = serial.Serial(
                    SERIAL_PORT,
                    BAUD_RATE,
                    timeout=1,
                    write_timeout=1,
                )
                time.sleep(2)
                print(f"Connected to ESP32 on {SERIAL_PORT}")
            except serial.SerialException as error:
                raise RuntimeError(
                    f"Could not open {SERIAL_PORT}. Check the port and close "
                    f"other programs using it. Error: {error}"
                ) from error
        else:
            print("Serial is disabled. No command sent to the arm.")
            show_serial_ports()

        candidate_gesture = None
        candidate_since = 0.0
        last_sent_gesture = None
        last_command_time = 0.0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                detected_gesture = "no hand"

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    detected_gesture = classify_gesture(hand)
                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )
                    wrist = hand.landmark[0]
                    x = int(wrist.x * width)
                    y = int(wrist.y * height)
                    cv2.putText(
                        frame,
                        detected_gesture,
                        (x, max(y - 20, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                now = time.monotonic()
                if detected_gesture != candidate_gesture:
                    candidate_gesture = detected_gesture
                    candidate_since = now

                stable = now - candidate_since >= STABLE_SECONDS
                if stable and detected_gesture in GESTURE_COMMANDS:
                    cooldown_passed = (
                        now - last_command_time >= COMMAND_COOLDOWN_SECONDS
                    )
                    if (
                        detected_gesture != last_sent_gesture
                        and cooldown_passed
                    ):
                        command = GESTURE_COMMANDS[detected_gesture]
                        send_command(connection, command)
                        last_sent_gesture = detected_gesture
                        last_command_time = now

                if stable and detected_gesture == "no hand":
                    last_sent_gesture = None

                mode_text = (
                    f"SERIAL ON: {SERIAL_PORT}"
                    if connection is not None
                    else "SERIAL OFF - recognition only"
                )
                cv2.putText(
                    frame,
                    mode_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255) if connection else (0, 255, 255),
                    2,
                )
                cv2.imshow("Gesture - press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if connection is not None:
            connection.close()
            print("Serial connection closed.")


if __name__ == "__main__":
    main()

