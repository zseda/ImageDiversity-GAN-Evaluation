import torch
import wandb
import imageio
from models.stylegan2 import Generator
from pathlib import Path


# Function to load the generator from a checkpoint
def load_generator(generator, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda"
    )  # Load the checkpoint on CUDA if available
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator


# Function to generate and save images
def generate_images(generator, num_images, image_size=32):
    with torch.no_grad():
        z = torch.randn(num_images, generator.z_dim, device=generator.device)
        c = torch.zeros(
            num_images, generator.c_dim, device=generator.device
        )  # Modify as needed for conditioning
        images = generator(
            z, c, img_resolution=image_size
        )  # Use the desired image size

    save_dir = Path("~/data/birinci/repo/generated_images").expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to numpy format
        img_path = save_dir / f"image_{i}.png"
        imageio.imsave(img_path, img)


# Function to log images to WandB
def log_images_to_wandb(images, num_images):
    wandb.init(project="stylegan2_generated_imgs", entity="independent_study")

    for i in range(0, num_images, 10):
        image_group = [
            images[j].cpu().numpy().transpose(1, 2, 0)
            for j in range(i, min(i + 10, num_images))
        ]
        captions = [f"Generated Image {j}" for j in range(i, min(i + 10, num_images))]
        wandb.log(
            {
                f"Generated Images {i}-{min(i + 10, num_images)}": [
                    wandb.Image(img, caption=captions[j])
                    for j, img in enumerate(image_group)
                ]
            }
        )

    wandb.finish()


def main():
    checkpoint_path = "model=G-best-weights-step=178000.pth"  # Replace with the actual path to your checkpoint
    num_images = 60000  # Change this to the number of images you want to generate
    image_size = 32  # Set the desired image size here (e.g., 32 for CIFAR-10)

    generator = Generator(
        z_dim=512, c_dim=0, w_dim=512, img_resolution=1024, img_channels=3, MODEL=None
    )
    generator = load_generator(generator, checkpoint_path)

    generate_images(generator, num_images, image_size)
    log_images_to_wandb(generator, num_images)


if __name__ == "__main__":
    main()
