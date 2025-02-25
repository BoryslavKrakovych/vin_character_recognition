import tensorflow as tf
import numpy as np
from PIL import Image
import os
import csv

model_path = "models/vin_model.h5"
model = tf.keras.models.load_model(model_path)

img_size = (28, 28)

input_dir = "data/emnist_test/"  
output_file = "output/results.csv"
os.makedirs("output", exist_ok=True)

if not os.path.exists(input_dir):
    exit(1)

results = []
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    img = Image.open(file_path).convert("L").resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  
    prediction = np.argmax(model.predict(img_array, verbose=0))
    results.append((prediction, file_path))
    print(f"Predicted: {prediction}, Path: {file_path}")

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Prediction", "Path"])
    writer.writerows(results)
