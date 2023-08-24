import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import time
from PIL import Image

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
dataloader_augmented = DataLoader(dataset_augmented, batch_size=32, shuffle=True)

# Load CIFAR-10 dataset with original transforms
dataset_original = CIFAR10(root="./data", train=True, download=True, transform=transform_original)
dataloader_original = DataLoader(dataset_original, batch_size=32, shuffle=True)

# Initialize lists to hold batched images
original_images_batch = []
augmented_images_batch = []

# Function to save and log image
def save_and_log_image(image, image_augmented, i, j):
    image_path = save_dir / f"{i}_{j}.png"
    transforms.ToPILImage()(image_augmented).save(image_path)
    
    # Load the image from the path as a PIL Image
    pil_image = Image.open(image_path)
    
    wandb.log({
        "Original Images": [wandb.Image(transforms.ToPILImage()(image), caption=f"Original {i}_{j}")],
        "Augmented Images": [wandb.Image(pil_image, caption=f"Augmented {i}_{j}")]
    })


# Log images to Wandb in batches
def log_images_to_wandb():
    wandb.log({
        "Original Images": original_images_batch,
        "Augmented Images": augmented_images_batch
    })
    original_images_batch.clear()
    augmented_images_batch.clear()

# Loop through datasets
for i, ((images, _), (images_augmented, _)) in enumerate(zip(dataloader_original, dataloader_augmented)):
    for j, (image, image_augmented) in enumerate(zip(images, images_augmented)):
        save_and_log_image(image, image_augmented, i, j)
    
    # Log images to Wandb in batches and rate-limit
    if i % 10 == 0:
        log_images_to_wandb()
        time.sleep(1)  # Rate-limit to avoid hitting system limits

# Log any remaining images to Wandb
if original_images_batch or augmented_images_batch:
    log_images_to_wandb()

# Log augmented images to wandb
wandb.save(str(save_dir / "*"))
