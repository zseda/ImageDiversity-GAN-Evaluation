# Use the base image with PyTorch and CUDA
FROM alex4727/experiment:pytorch113_cuda116

# Set the working directory
WORKDIR /app

# Install additional Python libraries using pip
RUN pip install pillow zipfile36 typer pathlib decouple

# Copy your source code to the container
COPY src/ /app

# Define the command to run your script
CMD ["python", "src/generate.py"]
