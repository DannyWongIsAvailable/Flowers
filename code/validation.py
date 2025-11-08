from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    # Load a model
    model = YOLO("../model/runs/classify/train/weights/best.pt")

    # Customize validation settings
    metrics = model.val(data="../datasets/flowers_cls")
    accuracy = metrics.top1
    print(f"Accuracy: {accuracy}")