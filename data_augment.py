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
dataloader_augmented = DataLoader(dataset_augmented, batch_size=1, shuffle=False)

# Load CIFAR-10 dataset with original transforms
dataset_original = CIFAR10(root="./data", train=True, download=True, transform=transform_original)
dataloader_original = DataLoader(dataset_original, batch_size=1, shuffle=False)

# Initialize counter
counter = 0

# Initialize lists to hold batch of images
original_images_batch = []
augmented_images_batch = []

# Function to save and log image
def save_and_log_image(image, image_augmented, counter):
    image_path = save_dir / f"{counter}.png"
    transforms.ToPILImage()(image_augmented).save(image_path)
    original_images_batch.append(wandb.Image(transforms.ToPILImage()(image), caption=f"Original {counter}"))
    augmented_images_batch.append(wandb.Image(image_path, caption=f"Augmented {counter}"))

# Loop through the dataset
for (image, _), (image_augmented, _) in zip(dataloader_original, dataloader_augmented):
    counter += 1
    save_and_log_image(image.squeeze(0), image_augmented.squeeze(0), counter)

    # Log every 10 images
    if counter % 10 == 0:
        wandb.log({
            "Original Images": original_images_batch,
            "Augmented Images": augmented_images_batch
        })
        # Clear the batch lists
        original_images_batch.clear()
        augmented_images_batch.clear()
        # Log any remaining images in the batch lists

if original_images_batch or augmented_images_batch:
    wandb.log({
        "Original Images": original_images_batch,
        "Augmented Images": augmented_images_batch
    })

# Save augmented images to directory
wandb.save(str(save_dir / "*"))

# Finish the Wandb run
wandb.finish()

