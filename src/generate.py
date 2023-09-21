import torch
import wandb
from PIL import Image
import zipfile
from models.stylegan2 import Generator
from pathlib import Path
import config as cfg
from settings import (
    CHECKPOINT_PATH,
    NUM_IMAGES,
    IMAGE_SIZE,
    ZIP_PATH,
    CFG_FILE,
    WANDB_API_KEY,
)


# Function to load the generator from a checkpoint
def load_generator(generator, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda"
    )  # Load the checkpoint on CUDA if available
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator


# Function to generate and save images in a zip file
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

    # Create a directory for saving generated images
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)

    # Create a zip file to store the images
    with zipfile.ZipFile(output_dir / zip_path, "w") as zipf:
        for i in range(num_images):
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to numpy format
            img = (img * 255).astype("uint8")  # Scale to 8-bit integer
            img = Image.fromarray(img)  # Convert to PIL Image
            img_data = img.tobytes()
            zipf.writestr(
                f"image_{i}.png", img_data
            )  # Save the image data to the zip file


# Function to log images to WandB
def log_images_to_wandb(images, num_images, wandb_api_key):
    wandb.init(
        project="generate synthetics images with stylegan2",
        entity="Image Diversity Gan Evaluation",
        config={"wandb_api_key": wandb_api_key},  # Pass the WandB API key in the config
    )

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


if __name__ == "__main__":
    cfgs = cfg.Configurations(CFG_FILE)
    generator = Generator(
        z_dim=512,
        c_dim=10,
        w_dim=512,
        img_resolution=32,
        img_channels=3,
        MODEL=cfgs.MODEL,
    )
    generator = load_generator(generator, CHECKPOINT_PATH)

    generate_and_zip_images(generator, NUM_IMAGES, IMAGE_SIZE, ZIP_PATH)
    log_images_to_wandb(generator, NUM_IMAGES, WANDB_API_KEY)
