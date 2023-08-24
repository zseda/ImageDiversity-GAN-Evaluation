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

# Log first 10 original and augmented images
original_images = next(iter(dataloader_original))[0]
augmented_images = next(iter(dataloader_augmented))[0]

wandb.log({
    "Original Images": [wandb.Image(transforms.ToPILImage()(img), caption=f"Original {i+1}") for i, img in enumerate(original_images)],
    "Augmented Images": [wandb.Image(transforms.ToPILImage()(img), caption=f"Augmented {i+1}") for i, img in enumerate(augmented_images)]
})

# Note on AutoAugment
wandb.log({"Note": "AutoAugment with CIFAR10 policy was used, but specific augmentations are not available."})

# Close the wandb run
wandb.finish()
