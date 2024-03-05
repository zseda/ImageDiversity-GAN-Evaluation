import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import wandb

# Initialize wandb
wandb.init(project="cifar10_class4_autoaugment")

# Log the AutoAugment policy used
wandb.config.update({"AutoAugment Policy": "CIFAR10, Class 4 Only"})

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create directory to save augmented images
save_dir = Path("CIFAR10_class4_augmented")
save_dir.mkdir(parents=True, exist_ok=True)

# Define AutoAugment transform for class 4
autoaugment_transform = transforms.AutoAugment(
    policy=transforms.AutoAugmentPolicy.CIFAR10
)

# Compose transforms for augmented images
transform_augmented = transforms.Compose(
    [
        autoaugment_transform,
        transforms.ToTensor(),
    ]
)


# Custom dataset to filter for class 4 only
class CIFAR10Class4(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        self.class4_indices = [
            i for i, (_, target) in enumerate(self.dataset) if target == 4
        ]

    def __len__(self):
        return len(self.class4_indices)

    def __getitem__(self, idx):
        original_idx = self.class4_indices[idx]
        return self.dataset[original_idx]


# Load CIFAR-10 dataset with augmented transforms for class 4
dataset_augmented = CIFAR10Class4(
    root="./data", train=True, download=True, transform=transform_augmented
)
dataloader_augmented = DataLoader(dataset_augmented, batch_size=10, shuffle=False)


# Function to save and log image
def save_and_log_image(image_augmented, idx):
    image_path = save_dir / f"{idx}.png"
    transforms.ToPILImage()(image_augmented).save(image_path)


# Initialize list to hold wandb.Image objects for augmented images
augmented_images = []

# Save augmented images and log to wandb
for i, (images_augmented, _) in enumerate(dataloader_augmented):
    for j, image_augmented in enumerate(images_augmented):
        save_and_log_image(image_augmented, i * 10 + j)
        if i % 10 == 0:  # Log every 10 batches
            augmented_images.append(
                wandb.Image(
                    transforms.ToPILImage()(image_augmented),
                    caption=f"Augmented Class 4 - {i*10 + j}",
                )
            )

    if i % 10 == 0:  # Log every 10 batches
        wandb.log({"Augmented Images": augmented_images})
        augmented_images = []

# Log augmented images to wandb
wandb.save(str(save_dir / "*"))
