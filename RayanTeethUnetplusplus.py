import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import re

# Model setup: UNet++ with EfficientNet-b0 backbone
def get_unetpp_model(num_classes):
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,  # Grayscale images
        classes=num_classes,
        activation=None  # Logits for loss calculation
    )
    return model

# Combined Dice and BCE Loss Function
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        bce_loss = self.bce(pred, target)
        return dice_loss + bce_loss

# Dataset definition with noise filtering
class TeethSegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose, dataset_type: str = 'Train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.dataset_type = dataset_type

        # Extract numeric part for sorting
        extract_number = lambda x: int(re.search(r'\d+', x).group())
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')], key=extract_number)
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')], key=extract_number)

        # Filter out noisy masks and their corresponding images
        self.images, self.masks = self.filter_noisy_masks(self.images, self.masks)

        # Split into train/test
        split_ratio = 0.8
        num_samples = len(self.images)
        if dataset_type == 'Train':
            self.images = self.images[:int(num_samples * split_ratio)]
            self.masks = self.masks[:int(num_samples * split_ratio)]
        elif dataset_type == 'Test':
            self.images = self.images[int(num_samples * split_ratio):]
            self.masks = self.masks[int(num_samples * split_ratio):]

    def filter_noisy_masks(self, images, masks):
        """Filter out noisy masks based on file size threshold."""
        filtered_images, filtered_masks = [], []
        size_threshold = 50 * 1024  # 50 KB in bytes

        for img, msk in zip(images, masks):
            mask_path = os.path.join(self.mask_dir, msk)
            if os.path.getsize(mask_path) <= size_threshold:
                filtered_images.append(img)
                filtered_masks.append(msk)

        return filtered_images, filtered_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = np.array(Image.open(image_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        transformed = self.transform(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']
        binary_mask = (mask > 0).float().unsqueeze(0)

        return image, binary_mask

# Augmentations with Albumentations
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0),
    ToTensorV2()
])
test_augmenter = A.Compose([A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0), ToTensorV2()])

# Dice Score Function
def dice_score(pred, target_mask, epsilon=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target_mask = target_mask.to(pred.device)  # Ensure both tensors are on the same device

    intersection = (pred * target_mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice.mean().item()

# Visualization
def visualize_mask(inputs, masks, outputs, num_samples=5):
    batch_size = inputs.size(0)
    samples_to_display = min(num_samples, batch_size)

    for i in range(samples_to_display):
        dice = dice_score(outputs[i:i+1], masks[i:i+1])
        print(f'Dice score for sample {i}: {dice}')

        inputs_np = inputs.cpu().numpy()
        masks_np = masks.cpu().numpy()
        outputs_np = outputs.detach().cpu().numpy()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(inputs_np[i].transpose(1, 2, 0), cmap='gray')
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks_np[i, 0], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(outputs_np[i, 0] > 0.5, cmap='gray')
        plt.title("Model Output Mask")
        plt.axis('off')

        plt.show()

# Check filtered masks
def check_filtered_masks(dataset, num_samples=5):
    for i in range(num_samples):
        mask_path = os.path.join(dataset.mask_dir, dataset.masks[i])
        mask = Image.open(mask_path)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Filtered Mask Sample {i+1}")
        plt.axis("off")
        plt.show()

# Training Function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss, dice_scores = 0.0, []

            for inputs, masks in dataloaders[phase]:
                inputs, masks = inputs.to(device), masks.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        dice_scores.append(dice_score(outputs, masks))

                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')
            if phase == 'test':
                print(f'Dice Score: {np.mean(dice_scores):.4f}')
                if scheduler:
                    scheduler.step(epoch_loss)

    # Visualize results after training completes
    inputs, masks = next(iter(dataloaders['test']))
    inputs, masks = inputs.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    visualize_mask(inputs, masks, outputs, num_samples=5)

    return model

# Main code
image_dir = '/content/drive/MyDrive/Main_teeth_dataset/images'
mask_dir = '/content/drive/MyDrive/Main_teeth_dataset/masks'
num_epochs =  50
batch_size = 4

train_dataset = TeethSegmentationDataset(image_dir, mask_dir, transform=augmenter, dataset_type='Train')
test_dataset = TeethSegmentationDataset(image_dir, mask_dir, transform=test_augmenter, dataset_type='Test')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
dataloaders = {"train": train_loader, "test": test_loader}

model = get_unetpp_model(num_classes=1)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, scheduler=scheduler)
torch.save(trained_model.state_dict(), "model.pth")
print("Training complete and model saved as model.pth")
