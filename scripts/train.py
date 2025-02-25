import os
import shutil
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_dir = "data/emnist_train/"
output_dir = "data/emnist_train_by_class/"  
model_output = "models/vin_model.h5"
batch_size = 32
img_size = (28, 28)

ascii_mapping = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

os.makedirs(output_dir, exist_ok=True)
for label in ascii_mapping:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    if file_name.endswith(".png"):
        label = file_name[0]  
        target_dir = os.path.join(output_dir, label)
        shutil.move(file_path, os.path.join(target_dir, file_name))


datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    output_dir, target_size=img_size, batch_size=batch_size, subset="training", 
    class_mode="categorical", color_mode="grayscale"
)

val_gen = datagen.flow_from_directory(
    output_dir, target_size=img_size, batch_size=batch_size, subset="validation", 
    class_mode="categorical", color_mode="grayscale"
)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(train_gen.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=10)
model.save(model_output)