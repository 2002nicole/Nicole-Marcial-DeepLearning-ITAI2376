# Requirements and Dependencies — L03 Convolutional Neural Networks Lab

This lab was designed to run in **Google Colab** or **Amazon SageMaker Studio Lab** with GPU acceleration. Google Colab was the recommended environment.

## Recommended Environment

```text
Python 3.10 or later
Google Colab recommended
GPU recommended
```

## Main Dependencies

```text
tensorflow
keras
numpy
matplotlib
scikit-learn
kaggle
pillow
seaborn
pandas
```

## Installation Command

The notebook installation cell used:

```bash
pip install -q tensorflow numpy matplotlib scikit-learn kaggle pillow seaborn
```

If running locally, install the required libraries with:

```bash
pip install tensorflow numpy matplotlib scikit-learn kaggle pillow seaborn pandas
```

## Frameworks and Libraries Used

### TensorFlow/Keras

TensorFlow/Keras was the main deep learning framework used in this lab. It was used to:

- Build a custom CNN
- Create the transfer learning model
- Train the models
- Fine-tune the transfer learning model
- Save the trained model as a `.keras` file

### ResNet50 or MobileNetV2

The course notebook allowed transfer learning with ResNet50 or MobileNetV2.

- **ResNet50** uses larger 224 × 224 images and requires more memory.
- **MobileNetV2** uses smaller 128 × 128 images and is lighter for Google Colab.

In my completed notebook, MobileNetV2 was used because it was more practical for the available runtime.

### ImageDataGenerator

`ImageDataGenerator` was used to:

- Load images from directories
- Rescale image pixel values
- Apply training data augmentation
- Create batches for model training

### Scikit-learn

Scikit-learn was used for model evaluation, including:

- Confusion matrix
- Classification report

### Matplotlib and Seaborn

Matplotlib and Seaborn were used to visualize:

- Sample images
- Training accuracy/loss curves
- Model predictions
- Confusion matrix

### Kaggle

Kaggle was included because the dataset comes from Kaggle. The notebook provides two options:

1. Download using Kaggle API credentials
2. Upload the dataset zip file manually

## Hardware Notes

A GPU is recommended because CNN training and transfer learning can be slow on CPU.

To enable GPU in Google Colab:

1. Go to **Runtime**
2. Select **Change runtime type**
3. Choose **GPU** as the hardware accelerator
4. Save and restart the runtime

You can verify GPU availability with:

```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices("GPU"))
```

## Memory Notes

If GPU memory issues occur, the notebook includes a lighter model option:

```python
USE_MOBILENET = True
```

Setting this to `True` switches from ResNet50 to MobileNetV2.

## Generated Files

Running the notebook may create the following files:

```text
best_custom_model.keras
best_transfer_model.keras
puppy_bagel_classifier.keras
```

These model files are generated outputs. They do not have to be committed to GitHub unless specifically needed for demonstration.
