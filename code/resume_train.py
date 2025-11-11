from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("../model/runs/classify/train2/weights/last.pt")
    model.train(resume=True)
