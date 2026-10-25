
import cv2
import mediapipe as mp
import math
import time
from pathlib import Path

# ==========================================
# SETTINGS
# ==========================================

CAMERA_INDEX = 0

# Keep serial disabled while testing.
# This program does not send motor commands.
ENABLE_SERIAL = False

STABLE_SECONDS = 0.7
COMMAND_COOLDOWN_SECONDS = 1.0

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "hand_landmarker.task"
)

GESTURE_COMMANDS = {
    "open palm": "OPEN",
    "fist": "FIST",
    "point": "POINT",
    "thumbs up": "THUMBS_UP",
    "peace": "PEACE",
}

# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def distance(a, b):
    """Calculate distance between two landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def joint_angle(a, b, c):
    """Calculate the angle at joint b."""

    ax = a.x - b.x
    ay = a.y - b.y

    cx = c.x - b.x
    cy = c.y - b.y

    dot = ax * cx + ay * cy

    magnitude = (
        math.hypot(ax, ay)
        * math.hypot(cx, cy)
    )

    if magnitude < 1e-8:
        return 0.0

    cosine = max(
        -1.0,
        min(1.0, dot / magnitude)
    )

    return math.degrees(math.acos(cosine))


# ==========================================
# IMPROVED GESTURE CLASSIFIER
# ==========================================

def classify_gesture(landmarks):
    """
    Recognise approved gestures using
    finger joint angles.
    """

    # MCP, PIP and TIP landmarks
    fingers = [
        (5, 6, 8),     # Index
        (9, 10, 12),   # Middle
        (13, 14, 16),  # Ring
        (17, 18, 20),  # Pinky
    ]

    extended = []

    for mcp, pip, tip in fingers:

        angle = joint_angle(
            landmarks[mcp],
            landmarks[pip],
            landmarks[tip]
        )

        # A straighter finger has a larger angle.
        is_extended = angle > 155
        extended.append(is_extended)

    index, middle, ring, pinky = extended

    # Check whether the thumb is extended.
    thumb_extended = (
        distance(landmarks[4], landmarks[17])
        >
        distance(landmarks[3], landmarks[17]) * 1.15
    )

    # PEACE SIGN
    if (
        index
        and middle
        and not ring
        and not pinky
    ):
        return "peace"

    # POINTING
    # Thumb can be folded or partially extended.
    if (
        index
        and not middle
        and not ring
        and not pinky
    ):
        return "point"

    # OPEN PALM
    if all(extended):
        return "open palm"

    # FIST OR THUMBS UP
    if not any(extended):

        if (
            thumb_extended
            and landmarks[4].y < landmarks[2].y
            and landmarks[4].y < landmarks[5].y
        ):
            return "thumbs up"

        if not thumb_extended:
            return "fist"

    # Unapproved or unclear gesture
    return "unknown"


# ==========================================
# DRAW HAND LANDMARKS
# ==========================================

def draw_hand(frame, landmarks):
    """Draw landmarks and connecting lines."""

    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    for point in points:

        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1
        )


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {MODEL_PATH}"
        )

    if ENABLE_SERIAL:
        raise RuntimeError(
            "Serial control is disabled in this test."
        )

    # MediaPipe Tasks API
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open laptop camera
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        camera.release()

        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}."
        )

    print("Gesture recognition started.")
    print("SERIAL OFF - Recognition only.")
    print("Press Q to quit.")

    candidate_gesture = None
    candidate_since = 0.0

    last_reported_gesture = None
    last_command_time = 0.0

    start_time = time.monotonic()
    last_timestamp_ms = -1

    try:

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                success, frame = camera.read()

                if not success:
                    print("Could not read camera frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time)
                    * 1000
                )

                timestamp_ms = max(
                    timestamp_ms,
                    last_timestamp_ms + 1
                )

                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                detected_gesture = "no hand"

                if result.hand_landmarks:

                    landmarks = result.hand_landmarks[0]

                    draw_hand(
                        frame,
                        landmarks
                    )

                    detected_gesture = classify_gesture(
                        landmarks
                    )

                # ----------------------------------
                # GESTURE STABILITY
                # ----------------------------------

                now = time.monotonic()

                if detected_gesture != candidate_gesture:

                    candidate_gesture = detected_gesture
                    candidate_since = now

                stable = (
                    now - candidate_since
                    >= STABLE_SECONDS
                )

                if detected_gesture == "no hand":
                    last_reported_gesture = None

                if (
                    stable
                    and detected_gesture
                    in GESTURE_COMMANDS
                ):

                    cooldown_passed = (
                        now - last_command_time
                        >= COMMAND_COOLDOWN_SECONDS
                    )

                    if (
                        detected_gesture
                        != last_reported_gesture
                        and cooldown_passed
                    ):

                        command = GESTURE_COMMANDS[
                            detected_gesture
                        ]

                        print(
                            "[SERIAL DISABLED] "
                            f"Recognised: {detected_gesture} "
                            f"-> Would send: {command}"
                        )

                        last_reported_gesture = (
                            detected_gesture
                        )

                        last_command_time = now

                # ----------------------------------
                # DISPLAY
                # ----------------------------------

                cv2.putText(
                    frame,
                    f"Gesture: {detected_gesture}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "SERIAL OFF - Recognition only",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(
                    "Hand Gesture Recognition",
                    frame
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    break

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("Camera closed safely.")


if __name__ == "__main__":
    main()
