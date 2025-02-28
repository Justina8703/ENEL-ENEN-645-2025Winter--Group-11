import os
import re
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
from transformers import DistilBertModel, DistilBertTokenizer

#############################################
# 1. Create a custom multi-modal dataset
#############################################

class MultiModalDataset(datasets.ImageFolder):
    """
    A custom dataset that extends ImageFolder to also extract text from the image file name.
    Each sample is returned as a dictionary with keys:
      - 'image': the transformed image tensor,
      - 'text': a string extracted from the file name,
      - 'label': the class label.
    """
    def __getitem__(self, index):
        # Get image path and label from the parent ImageFolder class
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        # Extract text from file name (remove extension, replace underscores, remove digits)
        filename = os.path.basename(path)
        filename_no_ext, _ = os.path.splitext(filename)
        text = filename_no_ext.replace('_', ' ')
        text = re.sub(r'\d+', '', text)  # remove digits
        return {"image": image, "text": text, "label": label}

#############################################
# 2. Setup directories 
#############################################


base_dir = '/work/TALC/enel645_2025w/garbage_data/'

if not os.path.exists(base_dir):
    base_dir = "/home/n-iznat/Desktop/garbage_data"

if not os.path.exists(base_dir):
    raise FileNotFoundError(f"dataset folder not found at: {base_dir}")
else:
    print("dataset folder exists:", base_dir)

# List folders in the dataset directory
drive_folders = os.listdir(base_dir)
print("Folders in dataset directory:", drive_folders)

# Define data directories
TRAIN_DIR = os.path.join(base_dir, "CVPR_2024_dataset_Train")
VAL_DIR   = os.path.join(base_dir, "CVPR_2024_dataset_Val")
TEST_DIR  = os.path.join(base_dir, "CVPR_2024_dataset_Test")

# If validation or test directories are missing, use TRAIN as a fallback.
for dir_name, dir_path in zip(["Train", "Val", "Test"], [TRAIN_DIR, VAL_DIR, TEST_DIR]):
    if not os.path.exists(dir_path):
        print(f"WARNING: {dir_name} directory not found at: {dir_path}")
        if dir_name in ["Val", "Test"]:
            print(f"Using TRAIN directory for {dir_name} to avoid errors.")
            if dir_name == "Val":
                VAL_DIR = TRAIN_DIR
            else:
                TEST_DIR = TRAIN_DIR

#############################################
# 3. Define image transformations
#############################################

transformations = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

#############################################
# 4. Create datasets and DataLoaders
#############################################

dataset_train = MultiModalDataset(TRAIN_DIR, transform=transformations["train"])
dataset_val = MultiModalDataset(VAL_DIR, transform=transformations["val"])
dataset_test = MultiModalDataset(TEST_DIR, transform=transformations["test"])

dataloaders = {
    "train": DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True),
    "val": DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=2, pin_memory=True),
    "test": DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=2, pin_memory=True),
}

#############################################
# 5. Define the Multi-Modal Classifier
#############################################

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(MultiModalClassifier, self).__init__()
        # Text encoder: DistilBERT
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Image encoder: MobileNetV2; remove classifier so it returns a feature vector of size 1280.
        self.image_model = models.mobilenet_v2(pretrained=True)
        self.image_model.classifier = nn.Identity()
        # Fusion: Concatenate text (768) and image (1280) features = 2048 dims.
        self.image_sequential = nn.Sequential(
            nn.Linear(1280, 750),
            nn.BatchNorm1d(750),

        )
        self.text_sequential = nn.Sequential(
            nn.Linear(768, 750),
            nn.BatchNorm1d(750)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, image, text):
        # Process image
        img_feat1 = self.image_model(image)  # (batch, 1280)
        img_feat = self.image_sequential(img_feat1)
        # Process text: tokenize and feed through DistilBERT.
        encoding = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        # Move tokens to the same device as image
        input_ids = encoding['input_ids'].to(image.device)
        attention_mask = encoding['attention_mask'].to(image.device)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the representation of the first token
        text_feat1 = text_outputs.last_hidden_state[:, 0]  # (batch, 768)
        text_feat = self.text_sequential(text_feat1)
        # Fuse features
        fused = torch.cat([img_feat, text_feat], dim=1)  # (batch, 2048)
        logits = self.classifier(fused)
        return logits

#############################################
# 6. Setup device, loss, and optimizer
#############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalClassifier(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

#############################################
# 7. Define the training loop with logging (without AMP)
#############################################

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_loss = 1e6
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            for i, batch in enumerate(dataloaders[phase]):
                images = batch['image'].to(device)
                texts = batch['text']  # list of strings
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, texts)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i % 10 == 0:
                    print(f"Phase '{phase}' - Batch {i} processed")
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), "best_multimodal_model.pth")
        print(f"End of Epoch {epoch+1}")
    print(f"\nBest Val Accuracy: {best_loss:.4f}")
    return model

#############################################
# 8. Train the Multi-Modal Model
#############################################

EPOCHS = 10
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=EPOCHS)


#############################################
# 9. Evaluate the Model: Classification Report, Confusion Matrix, Accuracy, and F1 Score
#############################################
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    batch_count = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)
            outputs = model(images, texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            batch_count += 1
            if batch_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {batch_count} batches, elapsed time: {elapsed:.2f} seconds")
    return np.array(all_labels), np.array(all_preds)

# Load the best model
model.load_state_dict(torch.load("best_multimodal_model.pth"))
model.eval()

# Evaluate on the test set
true_labels, pred_labels = evaluate_model(model, dataloaders['test'], device)

accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average='weighted')
print("Test Accuracy: {:.4f}".format(accuracy))
print("Test F1 Score: {:.4f}".format(f1))
print("Classification Report:\n", classification_report(true_labels, pred_labels, target_names=dataset_train.classes))

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=dataset_train.classes, yticklabels=dataset_train.classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("confMat.png")
plt.close()

#############################################
# 10. Visualize a few test results
#############################################
import random

def imshow(inp, title=None):
    """Display a tensor as an image (unnormalizing it first)."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

# Randomly select a few test samples to display
num_samples = 3
indices = random.sample(range(len(dataset_test)), num_samples)

counter = 1
for idx in indices:
    sample = dataset_test[idx]
    image = sample["image"]
    text = sample["text"]
    label = sample["label"]
    # Prepare image for model (add batch dimension)
    input_image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_image, [text])
        _, preds = torch.max(outputs, 1)
    pred_class = dataset_train.classes[preds.item()]
    true_class = dataset_train.classes[label]
    plt.figure()
    imshow(image, title=f"True: {true_class} | Pred: {pred_class}\nText: {text}")
    plt.savefig("test_"+str(counter)+".png")
    plt.close()
    counter += 1