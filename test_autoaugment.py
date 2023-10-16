import wandb
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from torchvision.utils import make_grid
from PIL import ImageDraw, ImageFont
import csv

# Initialize WandB
wandb.init(project="autoaugment_cifar10")

# Load CIFAR10 Data
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=5, shuffle=False
)  # set shuffle to False


# Extracting the AutoAugment policy details
def extract_policy_details(policy):
    return [
        (operation, magnitude)
        for operation, magnitude, prob in policy
        if prob and random.random() < prob
    ]


autoaugment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)

font = ImageFont.load_default()


# Open CSV for writing
with open("augmentation_details.csv", "w", newline="") as csvfile:
    fieldnames = ["Filename", "Policy_1", "Magnitude_1", "Policy_2", "Magnitude_2"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # writes the headers

    for b, (images, _) in enumerate(trainloader):
        if b * 5 >= 100:  # Only process the first 100 images
            break

        pil_images = [transforms.ToPILImage()(img) for img in images]
        augmented_images = [autoaugment(img) for img in pil_images]

        # Extracting policies and tagging
        policies = [
            autoaugment.policies[random.randint(0, len(autoaugment.policies) - 1)]
            for _ in pil_images
        ]
        operations_list = [extract_policy_details(policy) for policy in policies]
        for img, operations in zip(augmented_images, operations_list):
            tag = ", ".join([f"{op[0]}: {op[1]}" for op in operations])
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), tag, font=font, fill="white")

        original_grid = make_grid(images, nrow=5).cpu()
        augmented_grid = make_grid(
            [transforms.ToTensor()(img) for img in augmented_images], nrow=5
        ).cpu()

        wandb.log(
            {
                f"Original_Batch_{b+1}": [wandb.Image(original_grid)],
                f"Augmented_Batch_{b+1}": [wandb.Image(augmented_grid)],
            }
        )

        for i, operations in enumerate(operations_list):
            writer.writerow(
                {
                    "Filename": f"batch_{b+1}_img_{i+1}",
                    "Policy_1": operations[0][0] if len(operations) > 0 else None,
                    "Magnitude_1": operations[0][1] if len(operations) > 0 else None,
                    "Policy_2": operations[1][0] if len(operations) > 1 else None,
                    "Magnitude_2": operations[1][1] if len(operations) > 1 else None,
                }
            )

# Finalize WandB Logging
wandb.finish()
