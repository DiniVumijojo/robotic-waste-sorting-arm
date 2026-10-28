import time
import cv2
import serial
from ultralytics import YOLO

PORT = "COM7"
BAUD = 115200
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
except Exception:
    ser = None

VALID_CATEGORIES = {"plastic", "paper", "metal", "glass", "electronic"}
HAZARD_CATEGORIES = {"glass", "electronic"}

CATEGORY_COMMANDS = {
    "plastic": b"P",
    "paper": b"A",
    "metal": b"M",
    "glass": b"G",
    "electronic": b"E",
}

def main():
    model = YOLO("best.pt")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    stable_seconds = 1.0
    candidate_category = None
    candidate_since = 0.0
    last_reported_category = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
                
            results = model.predict(frame, conf=0.30, verbose=False)[0]
            display = frame.copy()
            best_detection = None

            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id].lower()
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if label in HAZARD_CATEGORIES:
                        color = (0, 0, 255)
                    elif label in VALID_CATEGORIES:
                        color = (0, 200, 0)
                    else:
                        color = (0, 150, 255)

                    text = f"{label}: {confidence*100:.1f}%"
                    if label in HAZARD_CATEGORIES:
                        text += " [HAZARD]"

                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if label in VALID_CATEGORIES:
                        if best_detection is None or confidence > best_detection[1]:
                            best_detection = (label, confidence)

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
                last_reported_category = candidate_category
                
                if ser and candidate_category in CATEGORY_COMMANDS:
                    cmd = CATEGORY_COMMANDS[candidate_category]
                    ser.write(cmd)

            if detected_category is None:
                last_reported_category = None

            if candidate_category in HAZARD_CATEGORIES:
                status = f"HAZARD WARNING: {candidate_category.upper()}"
            elif candidate_category:
                status = f"Sorting Target: {candidate_category.upper()}"
            else:
                status = "Scanning items..."

            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Recycling Classifier", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if ser:
            ser.close()

if __name__ == "__main__":
    main()