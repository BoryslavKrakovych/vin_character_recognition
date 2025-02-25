import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import datasets, transforms
import string

emnist_url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
output_dir = "data/"
train_output = os.path.join(output_dir, "emnist_train/")
test_output = os.path.join(output_dir, "emnist_test/")
os.makedirs(train_output, exist_ok=True)
os.makedirs(test_output, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_data = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

X_train, y_train = train_data.data.numpy(), train_data.targets.numpy()
X_test, y_test = test_data.data.numpy(), test_data.targets.numpy()

ascii_mapping = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

for i, (img, label) in enumerate(zip(X_train, y_train)):
    label_char = ascii_mapping[label]
    img = Image.fromarray(img.astype(np.uint8), mode="L")  
    img.save(os.path.join(train_output, f"{label_char}_{i}.png"))

for i, (img, label) in enumerate(zip(X_test, y_test)):
    label_char = ascii_mapping[label]
    img = Image.fromarray(img.astype(np.uint8), mode="L") 
    img.save(os.path.join(test_output, f"{label_char}_{i}.png"))
