# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'/root/part2/models/yolov10-main/yolov10-main/ultralytics/cfg/models/v8/yolov8s_focus.yaml')
    # model.load('yolov8s.pt')
    model.train(data=r'./ultralytics/cfg/datasets/RDD2022.yaml',
                imgsz=640,
                epochs=150,
                batch=64,
                workers=16,
                # device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                sr=False,
                )
