from ultralytics import YOLO

if __name__ == '__main__':

    # ------------------------------------------------------------
    # 1. 加载模型：根据 YAML 构建模型 + 加载 YOLO11x-cls 预训练权重
    # ------------------------------------------------------------
    model = YOLO("yolo11-Flower-cls.yaml", task="classify").load("../model/yolo11x-cls.pt")

    # ------------------------------------------------------------
    # 2. 开始训练
    # ------------------------------------------------------------
    results = model.train(
        data="../datasets/flowers_cls",
        epochs=200,
        imgsz=224,
        batch=32,

        # ------- 优化器 -------
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # ------- 学习率策略 -------
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,      # 余弦退火

        # ------- 数据增强（针对花卉优化） -------
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,

        # ------- 正则化 -------
        label_smoothing=0.1,

        # ------- 其他 -------
        project="flower_runs",
        name="yolo11m_flower_advanced",
    )
