import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import matplotlib.pyplot as plt
import random
from torchvision.utils import make_grid

# Dynamically compute number of classes from Task 1 , we have 81 class including background
n_classes = 81

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=n_classes, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)




class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()

# Step 1: Get sorted file paths
img_dir = "/content/processed_dataset/images"
mask_dir = "/content/processed_dataset/masks"

image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
mask_paths = sorted([os.path.join(mask_dir, f.replace('.jpg', '.png')) for f in os.listdir(img_dir)])

# Step 2: Train/Val Split
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# Step 3: Transforms
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Step 4: Dataset and Dataloaders
train_dataset = SegmentationDataset(train_imgs, train_masks, transform=train_transform)
val_dataset = SegmentationDataset(val_imgs, val_masks, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)




# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=81).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Metrics
iou_metric = MulticlassJaccardIndex(num_classes=81, average='macro').to(device)
pixel_acc = MulticlassAccuracy(num_classes=81, average='macro').to(device)

# TensorBoard logging
writer = SummaryWriter(log_dir="./runs/segmentation")

# Checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Training loop , we can keep epochs = 35 - 50 for 3K - 8K sampling dataset
def train_model(model, train_loader, val_loader, epochs=20):
    best_val_iou = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_iou = 0
        train_acc = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)

            train_iou += iou_metric(preds, masks).item()
            train_acc += pixel_acc(preds, masks).item()

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        val_acc = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_iou += iou_metric(preds, masks).item()
                val_acc += pixel_acc(preds, masks).item()

        # Epoch averages
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_acc /= len(val_loader)

        # Logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("IoU/Train", train_iou, epoch)
        writer.add_scalar("IoU/Val", val_iou, epoch)
        writer.add_scalar("PixelAcc/Train", train_acc, epoch)
        writer.add_scalar("PixelAcc/Val", val_acc, epoch)

        print(f"\n Epoch {epoch+1}: Train IoU={train_iou:.4f} | Val IoU={val_iou:.4f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), f"checkpoints/best_model_epoch_{epoch+1}.pth")
            print(" Best model saved.")

# Start training
train_model(model, train_loader, val_loader, epochs=20)




# Load best model (optional but good practice)
import glob

checkpoint_path = sorted(glob.glob("/content/coco_sample/checkpoints/*.pth"))[-1]
print(f"Loading checkpoint: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


# Get 5 random samples from validation set
samples = random.sample(range(len(val_dataset)), 5)

fig, axs = plt.subplots(5, 3, figsize=(12, 20))

for i, idx in enumerate(samples):
    image, true_mask = val_dataset[idx]
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(image_input).argmax(dim=1).squeeze().cpu()

    # Convert image tensor back to numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # Unnormalize
    img_np = np.clip(img_np, 0, 1)

    axs[i, 0].imshow(img_np)
    axs[i, 0].set_title("Input Image")
    axs[i, 0].axis('off')

    axs[i, 1].imshow(true_mask.cpu(), cmap='gray')
    axs[i, 1].set_title("Ground Truth")
    axs[i, 1].axis('off')

    axs[i, 2].imshow(pred_mask, cmap='gray')
    axs[i, 2].set_title("Predicted Mask")
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()
