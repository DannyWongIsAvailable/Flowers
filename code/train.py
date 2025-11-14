from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO("yolo12xx-cls-elite.yaml", task="classify").load("../model/yolo11x-cls.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="../datasets/flowers_cls", epochs=300, imgsz=160, batch=8)