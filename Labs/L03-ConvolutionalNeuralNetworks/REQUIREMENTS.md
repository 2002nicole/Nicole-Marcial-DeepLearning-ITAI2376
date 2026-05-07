# Requirements and Dependencies — L03 Convolutional Neural Networks Lab

This lab was designed to run in **Google Colab** with GPU acceleration. Google Colab already includes many of the required machine learning libraries, but the notebook also includes an installation cell.

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

If running locally, you can install the required libraries with:

```bash
pip install tensorflow numpy matplotlib scikit-learn kaggle pillow seaborn pandas
```

## Frameworks and Libraries Used

### TensorFlow/Keras

TensorFlow/Keras was the main deep learning framework used in this lab. It was used to:

- Build a custom CNN
- Create the MobileNetV2 transfer learning model
- Train the models
- Fine-tune the transfer learning model
- Save the trained model as a `.keras` file

### MobileNetV2

MobileNetV2 was used as the pre-trained transfer learning model. It was selected because it is lightweight and works well in Google Colab.

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

A GPU is recommended because CNN training can be slower on CPU.

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

## Generated Files

Running the notebook may create the following files:

```text
best_custom_model.keras
best_transfer_model.keras
puppy_bagel_classifier.keras
```

These model files are generated outputs. They do not have to be committed to GitHub unless specifically needed for demonstration.
