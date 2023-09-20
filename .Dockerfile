# Use the base image with PyTorch and CUDA
FROM alex4727/experiment:pytorch113_cuda116

# Set the working directory
WORKDIR /root/code

# Install additional Python libraries using pip
RUN pip install pillow zipfile36 typer pathlib

# Copy your source code to the container
COPY . /root/code

# Define the command to run your script
CMD ["python", "src/generate.py"]
