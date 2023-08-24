import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import wandb

# Initialize wandb
wandb.init(project="cifar10_autoaugment")

# Create directory to save augmented images
save_dir = Path("CIFAR10_augmented")
save_dir.mkdir(parents=True, exist_ok=True)

# Define AutoAugment transform
autoaugment_transform = transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)

# Compose transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    autoaugment_transform,
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Function to save image
def save_image(image, i, j):
    image_path = save_dir / f"{i}_{j}.png"
    transforms.ToPILImage()(image).save(image_path)

# Save augmented images and log to wandb
with ThreadPoolExecutor() as executor:
    for i, (images, _) in enumerate(dataloader):
        for j, image in enumerate(images):
            executor.submit(save_image, image, i, j)

# Log augmented images to wandb
wandb.save(str(save_dir / "*"))
