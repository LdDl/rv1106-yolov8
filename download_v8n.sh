# Not working cURL?
# curl https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt --create-dirs -o yolov8n.pt
python3 -c 'from ultralytics import YOLO; model = YOLO("yolov8n.pt"); model.export(format="onnx", opset=19, imgsz=320)'