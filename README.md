# VIN Character Recognition Project

## Description

The VIN Character Recognition project is designed to process and recognize vehicle identification number (VIN) characters from images. Using a convolutional neural network (CNN) model, the project accurately classifies both letters and numbers found in VINs. The CNN architecture consists of convolutional layers for feature extraction, followed by fully connected layers for classification.

The system outputs predictions in two formats: a CSV file, which includes the character index and corresponding image path, and the console, providing a quick overview of the results. To facilitate deployment, the project can be easily containerized with Docker, ensuring that all dependencies and the environment are properly set up for smooth operation.

## Author

*Boryslav Krakovych*

Student of the 4th year of Igor Sikorskyi KPI, Faculty of Applied Mathematics, specializing in Data Science and Mathematical Modeling.Over the past three years, I mastered the programming languages ​​Python, C, Java, SQL; and has developed skills in QA, machine learning, deep learning, reinforcement learning, and data analysis.

## Requirements

The project requires Python 3.10 and the following libraries:

- TensorFlow
- Torchvision
- scikit-learn
- Pillow
- OpenCV
- Numpy

## Project structure

- `data/` - folder for images (original and processed).
- `models/` - saved model after training.
- `output/` - inference results.
- `scripts/` - scripts for refinement, training and inference.

## Usage

1. Image processing:
   ```bash
   python scripts/preprocess_data.py
2. Model training:
   ```bash
   python scripts/train.py
3. Inference (example):
   ```bash
   python scripts/inference.py data/test_images

## Using Docker

1. Creating a Docker image:
   ```bash
   docker build -t emnist_project .
2. Starting the container:
   ```bash
   docker run -it --rm emnist_project bash
3. Checking the presence of files in the mounted directory:
   ```bash
   ls /app/data/emnist_test 
4. Checking the contents of the /app directory:
   ```bash
   docker run -it --rm emnist_project ls /app
5. Running a Python script for inference:
   ```bash
   docker run -it --rm emnist_project python3 /app/scripts/inference.py --input /mnt/
