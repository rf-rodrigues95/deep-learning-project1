import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

# Load labels
labels = pd.read_csv("./data/train_labels.csv")
image_folder = "./data/Train/"

# Encode labels
label_encoder = LabelEncoder()
labels["encoded_label"] = label_encoder.fit_transform(labels["label"])
num_classes = len(label_encoder.classes_)

# Display dataset info
print(f"Total images: {len(labels)}")
print(f"Number of classes: {num_classes}")
print(labels['label'].value_counts())

# Plot class distribution with actual Pokémon types
plt.figure(figsize=(12, 6))
sns.countplot(x=labels['label'], palette='viridis', order=labels['label'].value_counts().index)
plt.xlabel("Pokémon Type")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.title("Class Distribution of Pokémon Dataset by Type")
plt.show()

# Define dataset class
class PokemonDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.df.iloc[idx, 0]) + ".png")
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.df.iloc[idx, 1]  # Use Pokémon type instead of encoded label
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize dataset
dataset = PokemonDataset(labels, image_folder, transform=transform)

# Display sample images
def show_images(dataset, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img, label = dataset[i]
        print(dataset[i])
        img = img.permute(1, 2, 0).numpy()
        img = (img * 0.5) + 0.5  # Unnormalize
        axes[i].imshow(img)
        axes[i].set_title(f"Type: {label}")
        axes[i].axis("off")
    plt.show()

show_images(dataset)

# Check if all images have the same dimensions
image_shapes = [dataset[i][0].shape for i in range(len(dataset))]
unique_shapes = set(image_shapes)
print(f"Unique image shapes in dataset: {unique_shapes}")

# Compute image statistics
all_images = torch.stack([dataset[i][0] for i in range(len(dataset))])
mean = all_images.mean(dim=(0, 2, 3))
std = all_images.std(dim=(0, 2, 3))
print(f"Mean pixel values: {mean.numpy()}")
print(f"Standard deviation: {std.numpy()}")
