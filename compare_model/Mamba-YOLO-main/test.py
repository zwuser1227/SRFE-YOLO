
from ultralytics import YOLO
 
def main():
    # 加载模型，split='test'利用测试集进行测试
    model = YOLO(r"output_dir/RDD2022/mambayolo3/weights/best.pt")
    # model.val(data="data.yaml", split='test', imgsz=640, batch=32, device=0, workers=8)  #模型验证
    model.val(data="./ultralytics/cfg/datasets/data.yaml", split='test', imgsz=640, batch=16, device='', workers=8)  #模型验证
if __name__ == "__main__":
    main()
