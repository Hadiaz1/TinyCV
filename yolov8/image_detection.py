import cv2
from imread_from_url import imread_from_url
from PIL import Image
import sys
sys.path.append("/yolov8")


from detect import YOLOv8Detect

model_path = "models/yolov8n_facemask.onnx"

yolov8_detector = YOLOv8Detect(model_path, conf_threshold=0.3, iou_threhold=0.3)

# Input image
img_url = "https://callsam.com/wp-content/uploads/2019/12/crosswalk-featured.jpg"
img = imread_from_url(img_url)

# Detect objects
yolov8_detector(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img, data="fm")
cv2.imwrite("images/detected_objects_facemask.jpg", combined_img)
combined_img = Image.fromarray(combined_img)

combined_img.show()