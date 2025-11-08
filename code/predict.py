from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO("../models/runs/classify/train/weights/best.pt")  # load a custom model

    # Predict with the model
    results = model(r"../datasets/flowers_cls/val/Centaurea montana/img_011633.jpg")  # predict on an image
    # Process results list
    for result in results:
        probs = result.probs  # Probs object for classification outputs
        confidence = probs.top1conf  # Top 1 confidence
        category = result.names[probs.top1]
        print(f"{category}: {confidence:.2f}")
