import numpy as np
import tflite_runtime.interpreter as tflite
import os

# ✅ Make sure class order matches your training generator
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

# 🔁 Load TFLite model (cached for performance)
def load_tflite_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_leaf_model.tflite'))
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# 🔮 Predict using TFLite
def load_model_and_predict(image_array):
    interpreter = load_tflite_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize if needed (image should already be 128x128x3 float32 normalized)
    interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data, class_names
