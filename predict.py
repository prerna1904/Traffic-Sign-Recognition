import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

img_height, img_width = 64, 64
model_path = "models/traffic_sign_model.keras"

model = load_model(model_path)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return predicted_class


# Example usage
if __name__ == "__main__":
    test_image = "test.jpg"  # Put any test image here
    result = predict_image(test_image)
    print(f"Predicted Class: {result}")
