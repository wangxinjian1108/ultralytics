from ultralytics import YOLO
import os

# Load a model
model = YOLO("pths/yolo11x.pt")  # load an official model

# Export the model
model.export(format="onnx", dynamic=True, nms=True)
# add onnx check
os.system("onnxruntime-tools test yolo11x.onnx")
# yolo export model=yolo11n.pt format=onnx  # export official model
