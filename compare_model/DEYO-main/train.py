from ultralytics import RTDETR

# Load a model
model = RTDETR("yolov8-rtdetr.yaml")
# model.load("yolov8n.pt")

# Use the model
model.train(data = "data.yaml", 
            epochs = 150,
            batch=64,
            workers=16,
            optimizer='SGD',
            close_mosaic=10,
            resume=False,
            project='runs/train',
            name='exp',
            single_cls=False,
            cache=False,)

# # Eval the model
# model = RTDETR("DEYO-tiny.pt")
# model.val(data = "coco.yaml")  # for DEYO-tiny: 37.6 AP