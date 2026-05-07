# L03 — Convolutional Neural Networks Lab

## Lab Overview

This folder contains my completed Module 03 lab for **ITAI 2376: Deep Learning**. This was a course lab, not a fully independent project. The notebook included starter code and guided sections provided by the course. My work included completing the required coding sections, organizing the image dataset, training and evaluating models, completing the fine-tuning student challenge, and writing reflection/analysis responses.

The purpose of this lab was to practice **Convolutional Neural Networks (CNNs)** for binary image classification using the “Puppy or Bagel” dataset.

## Problem Statement

The task was to build an image classifier that could distinguish between two visually similar classes:

```text
Class 0: Bagel
Class 1: Puppy
```

This is a useful beginner computer vision task because curled-up puppies and bagels can share similar visual features, such as round shapes, brown/tan coloring, and textured surfaces. The goal was to understand how CNNs learn image features and why transfer learning can perform better than training a small CNN from scratch.

## Source Code

The main source file for this lab is:

```text
L03_Nicole_Marcial_ITAI2376.ipynb
```

The notebook includes:

- Google Colab/GPU setup
- Kaggle dataset download or manual dataset upload
- Dataset extraction and organization
- Image data loading with `ImageDataGenerator`
- Data augmentation
- Custom CNN model building
- Transfer learning with ResNet50 or MobileNetV2
- Model evaluation
- Prediction visualization
- Confusion matrix and classification report
- Five reflective questions
- Fine-tuning student challenge
- Challenge question response
- Model saving/export

## Course Lab Requirements Completed

This lab required completing four coding exercises in the custom CNN section:

1. Add the third convolutional block  
2. Add the fourth convolutional block  
3. Add the classification layers  
4. Compile the model  

The lab also required answering five reflective questions:

1. Why convolutional layers are better than fully connected layers for images
2. Why data augmentation was used
3. Why transfer learning often achieves better results
4. Which model performed better and why
5. What visual features make puppies look like bagels

The required student challenge was to fine-tune the transfer learning model by:

1. Unfreezing the base model
2. Freezing all layers except the last 30
3. Recompiling with a lower learning rate
4. Training for 5 additional epochs
5. Evaluating and comparing performance with the frozen model

## Dataset

This lab used the **Puppy or Bagel** dataset from Kaggle.

The dataset is intentionally small. In my notebook run, the dataset contained:

```text
Bagel images: 8
Puppy images: 8
Total images: 16
```

I organized the dataset into train, validation, and test folders:

```text
Training set:   10 images
Validation set: 2 images
Test set:       4 images
```

Because the dataset is very small, the results should be interpreted carefully. A model can show very high accuracy on the test set because the test set only contains 4 images.

## Approach and Methodology

### 1. Data Preparation

The dataset was extracted from a zip file and reorganized into a folder structure that Keras could read:

```text
puppy-or-bagel_organized/
├── train/
│   ├── bagel/
│   └── puppy/
├── validation/
│   ├── bagel/
│   └── puppy/
└── test/
    ├── bagel/
    └── puppy/
```

This organization was important because `flow_from_directory()` uses folder names as class labels.

### 2. Data Augmentation

The training images were augmented to help the model generalize better. Augmentation created slightly different versions of the images through transformations such as rotation, shifting, zooming, and horizontal flipping.

The training generator used:

```text
rescale = 1./255
rotation_range = 20
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True
```

Validation and test images were only rescaled, not augmented.

### 3. Custom CNN From Scratch

The first model was a custom CNN built with TensorFlow/Keras. The model used convolutional layers to detect image features and max pooling layers to reduce image size while keeping important patterns.

The custom CNN architecture was:

```text
Conv2D(32, 3x3, ReLU)
MaxPooling2D(2x2)

Conv2D(64, 3x3, ReLU)
MaxPooling2D(2x2)

Conv2D(128, 3x3, ReLU)
MaxPooling2D(2x2)

Conv2D(128, 3x3, ReLU)
MaxPooling2D(2x2)

Flatten
Dense(512, ReLU)
Dropout(0.5)
Dense(1, Sigmoid)
```

The model used:

```text
Optimizer: Adam
Loss: Binary crossentropy
Metric: Accuracy
Image size: 150 x 150
Batch size: 32
Planned epochs: 15
```

### 4. Transfer Learning

The second model used transfer learning with a pre-trained CNN model. The course notebook allowed either **ResNet50** or **MobileNetV2**. In my notebook, MobileNetV2 was used because it is lighter and works well in Google Colab when memory is limited.

The transfer learning model used:

```text
MobileNetV2 base model
GlobalAveragePooling2D
Dense(256, ReLU)
Dropout(0.5)
Dense(1, Sigmoid)
```

The base model was first frozen so that its pre-trained feature detectors would not be updated during the initial training step. This allowed the model to reuse patterns learned from a much larger image dataset, such as edges, curves, colors, textures, and shapes.

### 5. Fine-Tuning Challenge

For the required student challenge, I fine-tuned the transfer learning model by:

1. Unfreezing the MobileNetV2 base model
2. Freezing the earlier layers
3. Keeping the last 30 layers trainable
4. Recompiling the model with a lower learning rate
5. Training for 5 additional epochs

The fine-tuning learning rate was:

```text
Adam learning rate: 0.0001
```

## Results and Evaluation

### Custom CNN Results

The custom CNN trained successfully, but it did not perform well on the small test set.

```text
Custom CNN test accuracy: 50.00%
Custom CNN test loss: 0.6752
Training stopped after: 8 epochs
```

The custom CNN likely struggled because the dataset was extremely small. A CNN trained from scratch usually needs more image examples to learn useful visual patterns.

### Transfer Learning Results

The MobileNetV2 transfer learning model performed much better.

```text
MobileNetV2 test accuracy: 100.00%
MobileNetV2 test loss: 0.0021
Training epochs: 10
```

The classification report showed perfect precision, recall, and F1-score on the 4-image test set:

```text
Bagel precision: 1.00
Bagel recall:    1.00
Puppy precision: 1.00
Puppy recall:    1.00
Overall accuracy: 1.00
```

Because the test set only had 4 images, this result should not be treated as proof that the model would work perfectly on many new images. It mainly shows that transfer learning was much stronger than the custom CNN for this small lab dataset.

### Model Comparison

```text
Custom CNN:             50.00% test accuracy, 0.6752 test loss
Transfer MobileNetV2:  100.00% test accuracy, 0.0021 test loss
```

### Fine-Tuning Results

The fine-tuned model also reached:

```text
Fine-tuned accuracy: 100.00%
Fine-tuned loss: 0.0062
```

Fine-tuning did not clearly improve performance because the original transfer learning model was already at 100% test accuracy. The test loss increased slightly from 0.0021 to 0.0062, so there was no strong evidence that fine-tuning made the model better in this lab. This also showed why accuracy alone is not always enough for evaluation, especially when the test set is very small.

## Learning Outcomes

From this lab, I learned how CNNs are used for image classification and why they are better than fully connected networks for image data.

Key takeaways:

- CNNs preserve spatial relationships in images.
- Convolutional filters help detect visual patterns such as edges, textures, curves, and shapes.
- Max pooling reduces image size while keeping important features.
- Data augmentation helps reduce memorization by creating variation in the training images.
- Transfer learning is especially helpful when the dataset is small.
- Frozen pre-trained layers reuse general image features learned from a much larger dataset.
- MobileNetV2 performed much better than a custom CNN trained from scratch.
- Very small test sets can make results look stronger than they really are.
- Fine-tuning is not always useful if the transfer learning model is already performing well.
- Similar visual features, such as circular outlines and tan textures, can activate similar CNN feature maps and confuse a filter-based model.

## Notes About Authorship

This was a guided course lab with starter code provided as part of ITAI 2376. My contributions included completing the required CNN coding exercises, organizing the dataset into a working folder structure, training and evaluating the models, completing the transfer learning fine-tuning challenge, and writing the reflection and analysis responses.
