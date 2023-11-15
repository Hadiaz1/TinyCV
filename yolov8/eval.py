from ultralytics import YOLO
# from roboflow import Roboflow
# rf = Roboflow(api_key="IDRcLnAHO7DBlLqhSw5v")
# project = rf.workspace("tinycv").project("pedestrian-detection-cctv")
# dataset = project.version(1).download("yolov8")

def eval_model(model_path, data_yaml_path, imgsz):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path, imgsz=imgsz)
    return metrics

if __name__ == "__main__":
    eval_model("models/yolov8n_facemask_quantized.onnx", "data.yaml", 224)