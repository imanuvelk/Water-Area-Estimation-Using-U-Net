import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

# Global settings
height, width = 256, 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 45
BATCH_SIZE = 32
LR = 0.001
RATIO = 0.5
SAMPLE_NUM = 18
ENCODER = 'resnet50'
WEIGHTS = 'imagenet'
MODEL_SAVE_PATH = './best_model.pt'

# Dataset
class LoadData(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.len = len(images_path)
        self.transform = A.Compose([
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ])

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx]).convert('RGB')
        mask = Image.open(self.masks_path[idx]).convert('L')

        img, mask = np.array(img), np.array(mask)
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        img = np.transpose(img, (2, 0, 1)) / 255.0
        mask = np.expand_dims(mask, axis=0) / 255.0

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return self.len

# Model
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.arc = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)
        if masks is not None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1, loss2
        return logits

# Training function
def train_fn(data_loader, model, optimizer):
    model.train()
    total_diceloss = 0.0
    total_bceloss = 0.0

    for images, masks in tqdm(data_loader, desc="Training"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        logits, diceloss, bceloss = model(images, masks)

        loss = diceloss + bceloss
        loss.backward()
        optimizer.step()

        total_diceloss += diceloss.item()
        total_bceloss += bceloss.item()

    return total_diceloss / len(data_loader), total_bceloss / len(data_loader)

# Evaluation function
def eval_fn(data_loader, model):
    model.eval()
    total_diceloss = 0.0
    total_bceloss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits, diceloss, bceloss = model(images, masks)

            total_diceloss += diceloss.item()
            total_bceloss += bceloss.item()

    # Visualization
    image_batch, mask_batch = next(iter(data_loader))
    image = image_batch[SAMPLE_NUM]
    mask = mask_batch[SAMPLE_NUM]
    with torch.no_grad():
        logits_mask = model(image.unsqueeze(0).to(DEVICE))
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > RATIO).float()

    f, axarr = plt.subplots(1, 3, figsize=(12, 4))
    axarr[0].imshow(np.transpose(image.numpy(), (1, 2, 0)))
    axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray')
    axarr[2].imshow(np.squeeze(pred_mask.cpu().numpy()), cmap='gray')
    axarr[0].set_title("Image")
    axarr[1].set_title("Ground Truth")
    axarr[2].set_title("Prediction")
    plt.show()

    return total_diceloss / len(data_loader), total_bceloss / len(data_loader)

# Main execution
if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())

    # Replace with your local dataset paths
    X = sorted(glob.glob('./data/Images/*'))
    y = sorted(glob.glob('./data/Masks/*'))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    train_dataset = LoadData(X_train, y_train)
    valid_dataset = LoadData(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SegmentationModel().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    best_val_dice_loss = float('inf')
    best_val_bce_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_dice, train_bce = train_fn(train_loader, model, optimizer)
        val_dice, val_bce = eval_fn(valid_loader, model)

        print(f"Train Dice Loss: {train_dice:.4f}, BCE Loss: {train_bce:.4f}")
        print(f"Val Dice Loss:   {val_dice:.4f}, BCE Loss: {val_bce:.4f}")

        if val_dice < best_val_dice_loss or val_bce < best_val_bce_loss:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved.")
            best_val_dice_loss = val_dice
            best_val_bce_loss = val_bce

    # Final prediction on a sample
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    image_batch, mask_batch = next(iter(valid_loader))
    image = image_batch[SAMPLE_NUM]
    mask = mask_batch[SAMPLE_NUM]

    with torch.no_grad():
        logits_mask = model(image.unsqueeze(0).to(DEVICE))
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > RATIO).float()

    f, axarr = plt.subplots(1, 3, figsize=(12, 4))
    axarr[0].imshow(np.transpose(image.numpy(), (1, 2, 0)))
    axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray')
    axarr[2].imshow(np.squeeze(pred_mask.cpu().numpy()), cmap='gray')
    axarr[0].set_title("Image")
    axarr[1].set_title("Ground Truth")
    axarr[2].set_title("Prediction")
    plt.show()
