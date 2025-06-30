import tensorflow as tf

# Load the trained model (.h5 file)
model = tf.keras.models.load_model("potato_leaf_model.h5")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to file
with open("potato_leaf_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as potato_leaf_model.tflite")
