from decouple import config

CHECKPOINT_PATH = config("CHECKPOINT_PATH")
NUM_IMAGES = config("NUM_IMAGES", default=50000, cast=int)
IMAGE_SIZE = config("IMAGE_SIZE", default=32, cast=int)
ZIP_PATH = config("ZIP_PATH", default="generated_images.zip")
CFG_FILE = config("CFG_FILE", default="src/configs/CIFAR10/StyleGAN2.yaml")
WANDB_API_KEY = config("WANDB_API_KEY")
