import cv2

from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
CAMERA_INDEX = 1

def main():
    model = YOLO(MODEL_PATH)
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}. "
            "Try CAMERA_INDEX=1 if you have another camera."
        )
    print("Camera test running. Press Q to quit.")
    print("Model classes:", model.names)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read the frame from the camera.")
                break

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            cv2.imshow("Camera AI Test", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()