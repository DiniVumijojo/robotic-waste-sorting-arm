import cv2 
from ultralytics import YOLO
import serial
import time

SERIAL_PORT = "/dev/ttyUSB0" 
BAUD_RATE = 115200
ENABLE_SERIAL = False  
model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open the camera. Try changing VideoCapture(0) to VideoCapture(1)."
    )
connection = None
if ENABLE_SERIAL:
    try:
        connection = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to microcontroller on {SERIAL_PORT}")
    except Exception as e:
        print(f"Warning: Could not open serial port {SERIAL_PORT}: {e}")

previous_detection = set()

try:
    while True:
        ok, frame = camera.read()

        if not ok:
            print("Could not read the frame from the camera")
            break

        results = model.predict(frame, conf=0.45, verbose=False)
        result = results[0]
        current_detections = set()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])

                current_detections.add(label)

                if label not in previous_detection:
                    print(f"Detected: {label} ({confidence:.0%})")
                    if connection and connection.is_open:
                        connection.write(f"{label}\n".encode("utf-8"))

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (xyxy[0], max(xyxy[1] - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        previous_detection = current_detections

        cv2.imshow("Object Detected - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    if connection and connection.is_open:
        connection.close()
        print("Serial connection closed.")