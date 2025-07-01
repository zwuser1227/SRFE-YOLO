from ultralytics import YOLO
# from ultralytics import YOLOv10
def main():
    # 加载模型，split='test'利用测试集进行测试
    model = YOLO(r"runs/train/yolov8s_epoch150_rdd2022_CoordAtt/weights/best.pt")
    # model.val(data="data.yaml", split='test', imgsz=640, batch=32, device=0, workers=8)  #模型验证
    model.val(data="./ultralytics/cfg/datasets/RDD2022.yaml", split='test', imgsz=640, batch=16, device='', workers=8)  #模型验证
if __name__ == "__main__":
    main()
