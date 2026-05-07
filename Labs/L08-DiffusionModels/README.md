# L08 — Diffusion Models Lab

## Lab Overview

This folder contains my completed Module 08 lab for **ITAI 2376: Deep Learning**. This was a course lab, not a fully independent project. The notebook included starter code and guided implementation sections provided by the course. My work included completing the required code sections, training a class-conditional diffusion model, generating image samples, running CLIP-based evaluation, and writing a separate analysis report.

The purpose of this lab was to understand how diffusion models generate images by learning to add and remove noise step by step.

## Files Included

```text
L08_Notebook_Nicole_Marcial_ITAI2376.ipynb
L08_Report_Nicole_Marcial_ITAI2376.pdf
README.md
REQUIREMENTS.md
DATA_ACCESS.md
```

## Problem Statement

The goal of this lab was to build and train a diffusion model capable of generating new handwritten digit images from random noise. The model was trained on the MNIST dataset and conditioned on class labels, meaning it could be asked to generate a specific digit from 0 to 9.

The lab focused on understanding the full diffusion workflow:

- Adding Gaussian noise to real images during the forward diffusion process
- Training a U-Net to predict the noise that was added
- Running the reverse diffusion process to generate new images
- Conditioning the model on digit labels
- Evaluating generated image quality with CLIP

## Source Code and Report

### Notebook

The main source code file is:

```text
L08_Notebook_Nicole_Marcial_ITAI2376.ipynb
```

The notebook includes:

- PyTorch setup
- GPU/device check
- MNIST dataset loading
- Train/validation split
- DataLoader creation
- U-Net architecture components
- Time embedding
- Class conditioning
- Forward diffusion process
- Reverse diffusion process
- Training loop
- Generated image samples
- Denoising progression visualizations
- CLIP model setup and evaluation

### Report

The analysis report is:

```text
L08_Report_Nicole_Marcial_ITAI2376.pdf
```

The report answers the lab assessment questions and explains:

- What happens during forward diffusion
- Why noise is added gradually
- Why U-Net is useful for denoising
- Why skip connections matter
- How class conditioning works
- What the loss values show
- Why image quality did not fully match the numerical improvement
- What CLIP scores showed about the generated samples
- Practical applications, limitations, and future improvements

## Dataset

This lab used the **MNIST** handwritten digit dataset.

MNIST contains grayscale images of handwritten digits:

```text
Dataset: MNIST
Classes: 10 digits, 0–9
Image size: 28 × 28 pixels
Channels: 1 grayscale channel
Training samples: 60,000
Test samples: 10,000
```

In my notebook, the training data was split into:

```text
Training set:   48,000 images
Validation set: 12,000 images
Batch size:     64
```

The dataset was downloaded automatically using `torchvision.datasets.MNIST`, so no manual dataset upload is required.

## Approach and Methodology

### 1. Dataset Setup

The notebook loaded MNIST with PyTorch and normalized image values to the range `[-1, 1]`.

The selected configuration was:

```text
IMG_SIZE = 28
IMG_CH = 1
N_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 30
```

The notebook also checked the image shape, label shape, data types, and pixel value ranges before training.

### 2. Diffusion Process

The diffusion process used a 100-step noise schedule:

```text
n_steps = 100
beta_start = 0.0001
beta_end = 0.02
```

The forward diffusion process gradually added Gaussian noise to clean images. The reverse diffusion process started from random noise and used the trained U-Net to remove noise step by step.

### 3. U-Net Model Architecture

The model used a U-Net architecture because diffusion models need to predict pixel-level noise while keeping both overall structure and fine details.

The model included:

- `GELUConvBlock`
- `RearrangePoolBlock`
- `DownBlock`
- `UpBlock`
- `SinusoidalPositionEmbedBlock`
- `EmbedBlock`
- U-Net forward pass with skip connections
- Time conditioning
- Class conditioning

The model configuration was:

```text
Input resolution: 28 × 28
Input channels: 1
Time steps: 100
Condition classes: 10
Channel dimensions: 32, 64, 128
Total parameters: 1,521,921
Trainable parameters: 1,521,921
Estimated model memory: 5.81 MB
```

### 4. Time and Class Conditioning

The model was conditioned using both timestep information and digit labels.

- The timestep was encoded using sinusoidal position embeddings.
- The digit label was converted into a one-hot vector and passed through a class embedding block.
- These embeddings were added into the U-Net so the model knew both how much noise was present and which digit it was supposed to generate.

### 5. Training Loop

The training loop trained the U-Net to predict the exact noise added to each image.

For each training step:

1. Select a clean image from MNIST
2. Pick a random timestep
3. Add the corresponding amount of noise
4. Ask the U-Net to predict the added noise
5. Compare predicted noise to actual noise using mean squared error loss

The training setup used:

```text
Optimizer: Adam
Initial learning rate: 0.001
Weight decay: 1e-5
Loss function: Mean Squared Error
Gradient clipping: 1.0
Learning rate scheduler: ReduceLROnPlateau
Epochs: 30
```

### 6. Image Generation

After training, the model generated samples for all 10 digit classes. The generation process started with random noise and denoised the image over 100 reverse steps.

The notebook also visualized the generation process step by step to show how noise gradually became a digit-like image.

### 7. CLIP Evaluation

The notebook installed and loaded OpenAI CLIP using the `ViT-B/32` model. CLIP was used as an image-quality judge by comparing generated images to text descriptions.

CLIP evaluation was used to estimate whether generated samples looked like:

- a good handwritten digit
- a clear digit
- a blurry or poor-quality digit

## Results and Evaluation

### Training Results

The model trained for 30 epochs.

```text
Starting training loss:    0.1129
Final training loss:       0.0578
Best training loss:        0.0578
Training loss improvement: 48.8%
```

Validation loss also improved:

```text
Starting validation loss: 0.0792
Final validation loss:    0.0577
Best validation loss:     0.0572
```

The close training and validation losses suggested that training was stable and did not show strong overfitting.

### Generated Image Quality

Even though the loss improved, the final generated images were still visually weak. In my report, I noted that the model learned the noise-prediction task, but the generated images did not always look like clear handwritten digits.

This was an important evaluation point because the numerical loss improved more clearly than the visual quality.

### CLIP Evaluation Results

CLIP was successfully installed and loaded:

```text
CLIP model: ViT-B/32
Visual backbone: VisionTransformer
```

In one evaluation, CLIP recognized a large portion of generated digit 6 samples as good examples. However, I noted in my report that the CLIP results sometimes looked more positive than my human visual judgment.

Additional sample-level scoring showed:

```text
Highest quality example: Sample 9
Clear score: 93.5%
Blurry score: 4.7%

Weakest example: Sample 6
Clear score: 45.5%
Blurry score: 52.4%
```

This showed that CLIP can help rank generated outputs, but human review is still useful when image quality is visually weak.

## Learning Outcomes

From this lab, I learned how diffusion models generate images by learning to reverse a noise process.

Key takeaways:

- Diffusion models do not generate images in one step. They gradually denoise random noise.
- The U-Net acts as the denoiser by predicting the noise added at each timestep.
- Mean squared error loss measures how well the model predicts the added noise.
- Skip connections help preserve image details while the U-Net shrinks and rebuilds feature maps.
- Time embeddings help the model understand where it is in the denoising process.
- Class conditioning lets the model generate a specific digit instead of a random image.
- Lower loss does not always guarantee strong visual quality.
- CLIP can be used as an automatic evaluator, but it should not fully replace human visual judgment.
- A small MNIST diffusion model is useful for learning but not strong enough for realistic image generation.

## Limitations

The main limitations of this lab were:

- The model was trained only on MNIST.
- The generated digits were not always visually clear.
- Diffusion generation was slow because images were created step by step.
- The model was much smaller than real-world diffusion models.
- CLIP scores did not always match human visual judgment.
- More training, stronger sampling methods, or a larger model would likely be needed for better outputs.

## Future Improvements

If I continued developing this lab, I would improve it by:

1. Generating multiple samples per class and using CLIP to rank the best outputs.
2. Trying a faster or stronger sampler to improve image quality and generation speed.
3. Improving the conditioning process so generated digits match the intended class more clearly.
4. Training longer or experimenting with model size and noise schedule.
5. Comparing MNIST results with Fashion-MNIST or CIFAR-10 if more compute was available.

## Notes About Authorship

This was a guided course lab with starter code provided as part of ITAI 2376. My contributions included completing the required implementation sections, running the full notebook, debugging and validating training, generating class-conditioned samples, applying CLIP evaluation, and writing the separate analysis report.
