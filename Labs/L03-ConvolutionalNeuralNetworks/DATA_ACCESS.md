# Sample Data and Data Access — L03 Convolutional Neural Networks Lab

## Dataset Used

This lab uses the **Puppy or Bagel** dataset from Kaggle.

Dataset page:

```text
https://www.kaggle.com/datasets/returnofsputnik/puppy-or-bagel
```

The dataset contains images of puppies and bagels. The classification task is to predict whether an image belongs to the puppy class or the bagel class.

## Dataset Size Used in My Notebook

In my notebook run, the dataset contained:

```text
Bagel images: 8
Puppy images: 8
Total images: 16
```

After organizing the data, the split was:

```text
Training set:   10 images
Validation set: 2 images
Test set:       4 images
```

## Data Access Option 1 — Kaggle API

To download the dataset directly in Google Colab using the Kaggle API:

1. Go to your Kaggle account.
2. Open **Account Settings**.
3. Scroll to the API section.
4. Click **Create New Token**.
5. Download the `kaggle.json` file.
6. Upload `kaggle.json` in the notebook when prompted.
7. Run the Kaggle download command.

Example command:

```bash
kaggle datasets download -d returnofsputnik/puppy-or-bagel
unzip -q -o puppy-or-bagel.zip -d puppy-or-bagel
```

## Data Access Option 2 — Manual Upload

If the Kaggle API does not work, manually download the dataset zip file from Kaggle and upload it to Google Colab.

Steps:

1. Download the dataset from Kaggle.
2. Upload the zip file into Colab using the file browser.
3. Run the manual extraction cell in the notebook.

Example command:

```bash
unzip -q -o puppy-or-bagel.zip -d puppy-or-bagel
```

This is the method used in my completed notebook.

## Required Folder Structure

Keras `flow_from_directory()` expects images to be organized into folders by class. The notebook creates this organized structure:

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

Each class folder contains the corresponding image files.

## Should the Dataset Be Uploaded to GitHub?

No. The dataset does not need to be uploaded to GitHub. It is better to provide clear data access instructions because the data comes from Kaggle.

For this lab folder, include:

```text
L03_Nicole_Marcial_ITAI2376.ipynb
README.md
REQUIREMENTS.md
DATA_ACCESS.md
```

## Reproducibility Notes

To reproduce the lab:

1. Open the notebook in Google Colab.
2. Enable GPU if available.
3. Install the required libraries.
4. Download the Puppy or Bagel dataset from Kaggle or upload the zip manually.
5. Run the dataset extraction and organization cells.
6. Run all notebook cells from top to bottom.
7. Review the custom CNN results, transfer learning results, fine-tuning results, prediction visualizations, and confusion matrix.

## Important Limitation

The dataset used in this lab is very small. The 100% MobileNetV2 test accuracy was measured on only 4 test images, so it should be interpreted as a lab result rather than a production-ready model evaluation.
