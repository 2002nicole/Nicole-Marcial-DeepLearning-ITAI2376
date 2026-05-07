# Requirements and Dependencies — L08 Diffusion Models Lab

This lab was designed to run in **Google Colab** or an equivalent GPU-enabled environment. A GPU is strongly recommended because diffusion model training is computationally expensive.

## Recommended Environment

```text
Python 3.10 or later
Google Colab recommended
GPU recommended
Minimum GPU memory for MNIST: about 2 GB
```

In my notebook run, the environment used:

```text
GPU: Tesla T4
Available GPU memory: about 15 GB
Framework: PyTorch
```

## Main Dependencies

```text
torch
torchvision
numpy
matplotlib
einops
tqdm
ftfy
regex
Pillow
clip
```

## Installation Commands

The notebook installs `einops` with:

```bash
pip install einops
```

For CLIP evaluation, the notebook installs:

```bash
pip install -q ftfy regex tqdm
pip install -q git+https://github.com/openai/CLIP.git
```

If running locally, install the main requirements with:

```bash
pip install torch torchvision numpy matplotlib einops tqdm ftfy regex pillow
pip install git+https://github.com/openai/CLIP.git
```

## Libraries Used

### PyTorch

PyTorch was the main deep learning framework used in this lab. It was used for:

- Building the U-Net model
- Creating custom neural network blocks
- Implementing forward diffusion
- Implementing reverse diffusion
- Training with backpropagation
- Running the generation process

### Torchvision

Torchvision was used to:

- Download the MNIST dataset
- Apply image transforms
- Load image data into PyTorch

### Einops

Einops was used for tensor rearrangement, especially in the downsampling block.

### Matplotlib

Matplotlib was used to visualize:

- Forward noise progression
- Generated digit samples
- Training and validation loss curves
- Step-by-step denoising progress

### CLIP

OpenAI CLIP was used for optional image-quality evaluation. The notebook used the `ViT-B/32` CLIP model to compare generated images with text descriptions.

## Generated Files

Running the notebook may create files such as:

```text
best_diffusion_model.pt
best_diffusion_model.pt.backup
```

These are generated model checkpoint files. They do not have to be uploaded to GitHub unless needed for demonstration. If they are large, leave them out of GitHub and explain that the notebook can retrain or regenerate them.

## Hardware Notes

To enable GPU in Google Colab:

1. Go to **Runtime**
2. Select **Change runtime type**
3. Choose **GPU** as the hardware accelerator
4. Save and restart the runtime

You can verify GPU availability with:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## Troubleshooting Notes

If GPU memory errors occur:

- Reduce the batch size
- Use MNIST instead of a more complex dataset
- Reduce U-Net channel sizes
- Restart the runtime
- Clear CUDA memory with `torch.cuda.empty_cache()`

If training is unstable:

- Check the learning rate
- Check that images are normalized correctly
- Verify the noise schedule
- Monitor training and validation loss
- Use gradient clipping
