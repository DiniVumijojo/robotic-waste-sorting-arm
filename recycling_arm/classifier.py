import cv2
import time
from ultralytics import YOLO

VALID_CATEGORIES = {"plastic", "paper", "metal", "other"}

CATEGORY_WORDS = {
    "plastic": ("bottle", "plate", "spoon", "bag"),
    "paper": ("book", "pamphlets"),
    "metal": ("can",),
    "other": (),
}

def classify_label(label):
    label = label.lower()
    for category, words in CATEGORY_WORDS.items():
        if any(word in label for word in words):
            return category
    return "unknown"

def main():
    model = YOLO("yolo11n.pt")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera. Try changing VideoCapture(0) to VideoCapture(1).")

    stable_seconds = 1.0
    candidate_category = None
    candidate_since = 0.0
    last_reported_category = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a camera frame.")
                break
            results = model.predict(frame, conf=0.45, verbose=False)[0]
            display = frame.copy()
            best_detection = None

            if results.boxes is not None:
                for box in results.boxes:
                    label = model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    category = classify_label(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 200, 0) if category != "unknown" else (0, 150, 255)
                    text = f"{label}: {category}"

                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display,
                        text,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                    
                    if category in VALID_CATEGORIES:
                        if best_detection is None or confidence > best_detection[1]:
                            best_detection = (category, confidence)

            now = time.monotonic()
            detected_category = best_detection[0] if best_detection else None

            if detected_category != candidate_category:
                candidate_category = detected_category
                candidate_since = now

            if (
                candidate_category is not None
                and now - candidate_since >= stable_seconds
                and candidate_category != last_reported_category
            ):
                print(f"Stable category detected: {candidate_category}")
                last_reported_category = candidate_category

            if detected_category is None:
                last_reported_category = None

            status = (
                f"Candidate: {candidate_category}"
                if candidate_category
                else "No known sorting category detected"
            )
            cv2.putText(
                display,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Recycling classifier — press Q to quit", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()