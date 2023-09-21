from decouple import config

CHECKPOINT_PATH = config(
    "CHECKPOINT_PATH",
    default="~/data-fast2/birinci/checkpoints/model=G-best-weights-step=178000.pth",
)
NUM_IMAGES = config("NUM_IMAGES", default=50000, cast=int)
IMAGE_SIZE = config("IMAGE_SIZE", default=32, cast=int)
ZIP_PATH = config("ZIP_PATH", default="generated_images.zip")
CFG_FILE = config("CFG_FILE", default="src/configs/CIFAR10/StyleGAN2.yaml")
WANDB_API_KEY = config("WANDB_API_KEY")
OUTPUT_DIR = config("OUTPUT_DIR", default="~/data-fast2/birinci/data/generated_images")
