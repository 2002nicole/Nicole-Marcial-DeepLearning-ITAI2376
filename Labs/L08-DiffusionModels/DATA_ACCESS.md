# Sample Data and Data Access — L08 Diffusion Models Lab

## Dataset Used

This lab uses the **MNIST** handwritten digit dataset.

MNIST is a standard deep learning dataset containing grayscale images of handwritten digits from 0 to 9.

## Dataset Details

```text
Dataset: MNIST
Classes: 10 digits, 0–9
Image type: Grayscale
Image size: 28 × 28 pixels
Training images: 60,000
Test images: 10,000
```

## Dataset Split Used in My Notebook

The original MNIST training set was split into training and validation sets:

```text
Training set:   48,000 images
Validation set: 12,000 images
Batch size:     64
```

## How the Data Is Accessed

No manual dataset download is required.

The notebook downloads MNIST automatically using `torchvision.datasets.MNIST`.

Example:

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)
```

When the notebook runs for the first time, MNIST is downloaded into a local `data/` folder.

## Data Preprocessing

The notebook uses the following preprocessing:

```text
Convert image to tensor
Normalize grayscale pixel values to approximately [-1, 1]
Create 80/20 train-validation split
Load batches using PyTorch DataLoader
```

The notebook verified the batch shape and value ranges:

```text
Image batch shape: torch.Size([1, 1, 28, 28])
Image data type: torch.float32
Label data type: torch.int64
Min pixel value: -1.0
Max pixel value: about 1.0
```

## Dataset Options From the Lab

The course lab allowed students to choose one of the following datasets depending on compute resources:

```text
MNIST: Basic option, works on free Colab
Fashion-MNIST: Intermediate option
CIFAR-10: Advanced option, requires more GPU memory
```

My completed notebook used **MNIST**.

## Should the Dataset Be Uploaded to GitHub?

No. The MNIST dataset should not be uploaded to GitHub because it downloads automatically through Torchvision.

For this lab folder, include:

```text
L08_Notebook_Nicole_Marcial_ITAI2376.ipynb
L08_Report_Nicole_Marcial_ITAI2376.pdf
README.md
REQUIREMENTS.md
DATA_ACCESS.md
```

## Reproducibility Notes

To reproduce the lab:

1. Open the notebook in Google Colab.
2. Enable GPU.
3. Run the setup and package installation cells.
4. Let Torchvision download MNIST automatically.
5. Run all cells from top to bottom.
6. Review the training loss, generated samples, denoising visualizations, CLIP evaluation, and the PDF report.

## Important Limitation

This lab uses MNIST, which is useful for learning but much simpler than real-world image generation datasets. The model demonstrates the diffusion process, but it is not intended to produce production-quality images.
