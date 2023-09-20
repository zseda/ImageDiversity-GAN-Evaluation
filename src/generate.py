import os
import torch
import wandb
from PIL import Image
import zipfile
import typer
from stylegan2 import Generator  # Import your Generator class from stylegan2.py
from pathlib import Path  # Import pathlib for file and directory operations

app = typer.Typer()


# Function to load the generator from a checkpoint
def load_generator(generator, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda"
    )  # Load the checkpoint on CUDA if available
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator


# Function to generate and save images
def generate_and_zip_images(
    generator, num_images, image_size=32, zip_path="generated_images.zip"
):
    with torch.no_grad():
        z = torch.randn(num_images, generator.z_dim, device=generator.device)
        c = torch.zeros(
            num_images, generator.c_dim, device=generator.device
        )  # Modify as needed for conditioning
        images = generator(
            z, c, img_resolution=image_size
        )  # Use the desired image size

    # Define the directory to save generated images
    save_dir = Path.home() / "data" / "birinci" / "repo" / "generated_images"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create a zip file to store the images
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i in range(num_images):
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to numpy format
            img = (img * 255).astype("uint8")  # Scale to 8-bit integer
            img = Image.fromarray(img)  # Convert to PIL Image
            img_path = save_dir / f"image_{i}.png"
            img.save(img_path)  # Save the image
            zipf.write(img_path, os.path.basename(img_path))


# Function to log images to WandB
def log_images_to_wandb(images, num_images):
    wandb.init(project="your_project_name", entity="your_entity_name")

    for i in range(0, num_images, 10):
        image_group = [
            images[j].cpu().numpy().transpose(1, 2, 0)
            for j in range(i, min(i + 10, num_images))
        ]
        captions = [f"Generated Image {j}" for j in range(i, min(i + 10, num_images))]
        wandb.log(
            {
                f"Generated Images {i}-{min(i + 10, num_images)}": [
                    wandb.Image(data=img, caption=captions[j])
                    for j, img in enumerate(image_group)
                ]
            }
        )

    wandb.finish()


@app.command()
def main(
    checkpoint_path: str = "path/to/your/checkpoint.pth",  # Replace with the actual path to your checkpoint
    num_images: int = 100,  # Change this to the number of images you want to generate
    image_size: int = 32,  # Set the desired image size here (e.g., 32 for CIFAR-10)
    zip_path: str = "generated_images.zip",  # Path to the zip file
):
    generator = Generator(
        z_dim=512, c_dim=0, w_dim=512, img_resolution=1024, img_channels=3, MODEL=None
    )
    generator = load_generator(generator, checkpoint_path)

    generate_and_zip_images(generator, num_images, image_size, zip_path)
    log_images_to_wandb(generator, num_images)


if __name__ == "__main__":
    app()
