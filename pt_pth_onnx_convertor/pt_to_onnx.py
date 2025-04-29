from ultralytics import YOLO

# Convert best.pt (YOLOv8n-based) to ONNX
def convert_yolov8n_to_onnx():
    # Load the YOLOv8n model with best.pt
    model = YOLO('best.pt', task='detect')  # Load your best.pt file

    # Export to ONNX directly using Ultralytics' export method
    model.export(format='onnx', imgsz=640, opset=12, dynamic=True)
    print("YOLOv8n best.pt model converted to ONNX: best.onnx")

if __name__ == "__main__":
    convert_yolov8n_to_onnx()