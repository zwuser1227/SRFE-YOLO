from ultralytics import YOLOv10
yolo = YOLOv10("runs/yolov10_epoch200/weights/best.pt",task="detect")
result = yolo(source=r"/root/part2/datasets/RaodDatasets/images/test",save=True,save_conf=True,save_txt=True,name='output')