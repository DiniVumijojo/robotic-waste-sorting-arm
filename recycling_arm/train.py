from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    
    results = model.train(
        data=r"C:\Users\seopa\robotic-waste-sorting-arm\recycling_arm\dataset\images\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="recycling_detector"
    )
    print("Training complete! Custom weights saved.")

if __name__ == "__main__":
    main()
