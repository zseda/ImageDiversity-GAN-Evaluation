import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from pathlib import Path
import wandb

# Initialize wandb
wandb.init(project="cifar10_autoaugment")

# Log the AutoAugment policy used
wandb.config.update({"AutoAugment Policy": "CIFAR10"})

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create directory to save augmented images
save_dir = Path("CIFAR10_augmented")
save_dir.mkdir(parents=True, exist_ok=True)

# Define AutoAugment transform
autoaugment_transform = transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)

# Compose transforms for augmented images
transform_augmented = transforms.Compose([
    autoaugment_transform,
    transforms.ToTensor(),
])

# Compose transforms for original images
transform_original = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset with augmented transforms
dataset_augmented = CIFAR10(root="./data", train=True, download=True, transform=transform_augmented)
dataloader_augmented = DataLoader(dataset_augmented, batch_size=10, shuffle=False)

# Load CIFAR-10 dataset with original transforms
dataset_original = CIFAR10(root="./data", train=True, download=True, transform=transform_original)
dataloader_original = DataLoader(dataset_original, batch_size=10, shuffle=False)

# Function to save and log image
def save_and_log_image(image, image_augmented, i, j):
    image_path = save_dir / f"{i*10 + j}.png"
    transforms.ToPILImage()(image_augmented).save(image_path)

# Initialize lists to hold wandb.Image objects
original_images = []
augmented_images = []

# Save augmented images and log to wandb
for i, ((images, _), (images_augmented, _)) in enumerate(zip(dataloader_original, dataloader_augmented)):
    for j, (image, image_augmented) in enumerate(zip(images, images_augmented)):
        save_and_log_image(image, image_augmented, i, j)
        if i % 10 == 0:  # Log every 10 batches
            original_images.append(wandb.Image(transforms.ToPILImage()(image), caption=f"Original {i*10 + j}"))
            augmented_images.append(wandb.Image(transforms.ToPILImage()(image_augmented), caption=f"Augmented {i*10 + j}"))
    
    if i % 10 == 0:  # Log every 10 batches
        wandb.log({
            "Original Images": original_images,
            "Augmented Images": augmented_images
        })
        original_images = []
        augmented_images = []

# Log augmented images to wandb
wandb.save(str(save_dir / "*"))