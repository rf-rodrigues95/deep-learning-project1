import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import f1_score
import time

# Load data and labels
labels = pd.read_csv("./data/train_labels.csv")
image_folder = "./data/Train/"

label_encoder = LabelEncoder()
labels["label"] = label_encoder.fit_transform(labels["label"])

num_classes = len(label_encoder.classes_)

class PokemonDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.df.iloc[idx, 0]))

        if not img_name.endswith('.png'):
            img_name += ".png"

        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if len(self.df.columns) > 1:  # Train Set has labels, Test does not.
            label = self.df.iloc[idx, 1]
            return image, label
        else:  
            return image, -1  # X dont care for Test

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, layer1_size = 512, layer2_size = 256, dropout = 0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layer1_size)
        self.bn1 = nn.BatchNorm1d(layer1_size)
        self.fc2 = nn.Linear(layer1_size, layer2_size)
        self.bn2 = nn.BatchNorm1d(layer2_size)
        self.fc3 = nn.Linear(layer2_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  
        return x

# Define Transformations (Resize, Normalize, Flatten)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,)),  # Normalize
    transforms.Lambda(lambda x: x.view(-1))  # Flatten
])

# Split the dataset into 3 (70% train, 10% val, 20% test)
dataset = PokemonDataset(labels, image_folder, transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

sample_image, _ = dataset[0]
input_size = sample_image.numel()

# Choose single hyperparameter values
n_epochs = 50
batch_size = 16
learning_rate = 0.001
step_size = 5
l1_size = 128
l2_size = 128
dropout = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Loop
def training_loop():
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, optimizer, and scheduler
    model = MLP(input_size, num_classes, l1_size, l2_size, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    model.to(device)

    start_time = time.time()

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  # Get predicted class indices
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate F1 Score
        f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}, F1 Score: {f1:.4f}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Test loop
    model.eval()
    correct, total = 0, 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1 Score: {f1:.4f}")

    return model

# Train the model
trained_model = training_loop()

# Test folder path for predictions
test_folder = "./data/Test"

# Create the Test dataset (without labels)
test_files = os.listdir(test_folder)
test_files = [f for f in test_files if f.endswith('.png')]  # Assuming PNG format for test images

# Create a DataFrame to hold the test file names (Ids)
test_df = pd.DataFrame({'Id': test_files})  # Only filenames (Ids)

# Create PokemonDataset for test images (no labels)
test_dataset = PokemonDataset(test_df, test_folder, transform=transform)

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Predict on the test dataset
predictions = []
trained_model.eval()
with torch.no_grad():
    for images, _ in test_loader:  # No labels in test set
        images = images.to(device)
        outputs = trained_model(images)
        _, preds = torch.max(outputs, 1)  # Get predicted class indices
        predictions.extend(preds.cpu().numpy())

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'Id': [file.split('.')[0] for file in test_files],  # extract id
    'Category': label_encoder.inverse_transform(predictions)  
})

# Save the submission file
submission.to_csv("submit.csv", index=False)

print("Submission file saved as 'submission.csv'")
