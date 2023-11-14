from ultralytics import YOLO

def pt_to_onnx(model_path, imgsz=240, optimize=False, simplify=False, opset=None):
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=imgsz, optimize=optimize, simplify=simplify, opset=opset)

if __name__ == "__main__":
    pt_to_onnx(model_path="models/yolov8n_facemask.pt", imgsz=224)
