import onnxruntime
import cv2
import numpy as np
import os

from onnxruntime.quantization import quantize, quantize_static, QuantFormat, QuantType, CalibrationDataReader

class CalibReader(CalibrationDataReader):
    def __init__(self, calibration_images_dir: str, model_path: str):
        self.enum_data = None

        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert images to input data
        self.nhwc_data_list = _preprocess_images(calibration_images_dir, height, width, size_limit=0)

        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def _preprocess_images(imgs_path, height, width, size_limit=0):
    image_names = os.listdir(imgs_path)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names

    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = imgs_path + "/" + image_name
        # Read the image using OpenCV
        image = cv2.imread(image_filepath)

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

        unconcatenated_batch_data.append(input_tensor)

    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

def static_quantize_onnx(unquantized_model_path,
                         quantized_model_path,
                         calibration_data,
                         quant_format=QuantFormat.QDQ,
                         per_channel=True,
                         weight_type=QuantType.QInt8,
                         activation_type=QuantType.QInt8,
                         reduce_range=False):

    # onnxruntime.quantization.quant_pre_process(unquantized_model_path, quantized_model_path)
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

    onnxruntime.quantization.quant_pre_process(unquantized_model_path, quantized_model_path)
    # Quantize the model
    static_quantize_onnx(unquantized_model_path, quantized_model_path, calibration_data)

    print(f"Quantization completed. Quantized model saved to: {quantized_model_path}")
