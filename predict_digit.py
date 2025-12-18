import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

# Load image
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28,28))
img = img.reshape(1,28,28,1) / 255.0

# Predict
prediction = model.predict(img)
digit = np.argmax(prediction)

print("Predicted Digit:", digit)
