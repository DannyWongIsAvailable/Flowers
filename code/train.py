from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO("yolo11x-cls.yaml").load("../models/yolo11x-cls.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="../datasets/flowers_cls", epochs=300, imgsz=160, batch=128)