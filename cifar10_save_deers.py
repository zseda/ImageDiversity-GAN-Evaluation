import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from pathlib import Path
import PIL.Image as Image

# Define the directory where you want to save the images
save_dir = Path("CIFAR10_class4_training")
save_dir.mkdir(parents=True, exist_ok=True)

# Define the transform to convert Torch tensors to PIL images
transform_to_pil = transforms.ToPILImage()

# Load the CIFAR-10 training dataset
trainset = CIFAR10(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)

# Loop through the training dataset
for idx, (image, label) in enumerate(trainset):
    # Check if the current image is of class 4
    if label == 4:
        # Convert the tensor image to a PIL image
        pil_image = transform_to_pil(image)

        # Define the path for saving the image
        image_path = save_dir / f"class4_{idx}.png"

        # Save the image
        pil_image.save(image_path)

print(f"Saved all class 4 training images to {save_dir}")
