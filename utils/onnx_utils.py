import onnxruntime
import cv2
import numpy as np
import os

from onnxruntime.quantization import quantize, quantize_static, QuantFormat, QuantType, CalibrationDataReader

class CalibReader(CalibrationDataReader):
    def __init__(self, calibration_images_dir: str, model_path: str):
        self.data_paths = calibration_images_dir
        self.image_files = [os.path.join(calibration_images_dir, f) for f in os.listdir(calibration_images_dir) if
                            f.endswith(('.jpg', '.png', '.jpeg'))]
        session = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
        (_, _, self.height, self.width) = session.get_inputs()[0].shape

    def get_next(self):
        try:
            # Load the image and preprocess it
            img_path = self.image_files.pop(0)
            img_data = preprocess_image(img_path, self.height, self.width)

            # Return the input data as a dictionary with the input name
            return img_data
        except IndexError:
            # If no more data is available, return None
            return None


def preprocess_image(img_path, height, width):
    # Read the image using OpenCV
    image = cv2.imread(img_path)

    # Convert BGR to RGB
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the model's input size
    input_img = cv2.resize(input_img, (width, height))

    # Normalize pixel values to be in the range [0, 1]
    input_img = input_img / 255.0

    # Transpose the channels to match the ONNX model input format (HWC to CHW)
    input_img = input_img.transpose(2, 0, 1)

    # Add an extra dimension to represent the batch size (1 in this case)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor

def static_quantize_onnx(unquantized_model_path,
                         quantized_model_path,
                         calibration_data,
                         quant_format=QuantFormat.QDQ,
                         per_channel=True,
                         weight_type=QuantType.QInt8,
                         activation_type=QuantType.QInt8,
                         reduce_range=False):

    onnxruntime.quantization.quant_pre_process(unquantized_model_path, quantized_model_path)
    quantize_static(
        unquantized_model_path,
        quantized_model_path,
        calibration_data_reader=calibration_data,
        quant_format=quant_format,
        per_channel=per_channel,
        weight_type=weight_type,
        activation_type=activation_type,
        reduce_range=reduce_range)

if __name__ == "__main__":
    model_path = "models/yolov8n_facemask.onnx"
    calibration_images_dir = "data/valid/images"

    # Create an instance of the CalibReader class
    calibration_data = CalibReader(calibration_images_dir, model_path)


    # Perform static quantization
    unquantized_model_path = "models/yolov8n_facemask.onnx"
    quantized_model_path = "models/yolov8n_facemask_quantized.onnx"

    # Ensure that the unquantized model exists
    if not os.path.exists(unquantized_model_path):
        raise FileNotFoundError(f"The unquantized model {unquantized_model_path} does not exist. Please provide the path to an unquantized model.")

    # # Quantize the model
    # static_quantize_onnx(unquantized_model_path, quantized_model_path, calibration_data)
    #
    # print(f"Quantization completed. Quantized model saved to: {quantized_model_path}")
