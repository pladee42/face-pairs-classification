# Face Verification via Pair Matching

This repository implements a face verification system that classifies whether two images belong to the same person. The system uses data augmentation, Principal Component Analysis (PCA) for dimensionality reduction, and a stacking ensemble model of Support Vector Classifier (SVC), Multi-Layer Perceptron (MLP), and Gradient Boosting Classifier (GBC) to achieve high accuracy.

## Features

- **Face Verification:** Classifies whether two images represent the same person.
- **Data Augmentation:** Includes techniques like horizontal flips, brightness/contrast adjustments, sharpness, and Gaussian blur to improve generalization.
- **PCA (Principal Component Analysis):** Reduces dimensionality of the input data for improved performance and efficiency.
- **Model Stacking:** Combines predictions from SVC, MLP, and GBC classifiers using a Logistic Regression meta-classifier.
- **Cross-Validation & Hyperparameter Tuning:** Optimizes hyperparameters using `RandomizedSearchCV` with 5-fold cross-validation.
- **Model Evaluation:** Evaluates the trained model on a separate test dataset (`eval1.joblib`).

## Prerequisites

Before running the code, ensure that the following Python packages are installed. You can install them using the provided `requirements.txt` file.

### Installing Dependencies

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   ```

2. Activate the virtual environment:

   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Alternatively, you can install the dependencies directly using `pip`:

```bash
pip install numpy scikit-learn Pillow joblib
```

## Usage

### Training the Model

The script `train.py` processes the input data, applies augmentation, performs model training, and saves the trained model pipeline to a file. The following arguments can be passed when running the script:

#### Arguments

- `training_data` (str): Path to the `.joblib` file containing the training data dictionary (with keys `'data'` and `'target'`).
- `model_file_name` (str): The file name to save the trained model pipeline (default: `'model.joblib'`).
- `--test-aug` (optional): If set, will search for the best augmentation ratio.
- `--test-pca` (optional): If set, will search for the best PCA components.

#### Example

```bash
python train.py train.joblib model.joblib --test-aug --test-pca
```

This command will train the model using the `train.joblib` dataset, perform augmentation ratio and PCA component searches, and save the resulting pipeline as `model.joblib`.

### Evaluating the Model

The `evaluate.py` script is used to evaluate the performance of a trained model on a separate dataset (`eval1.joblib`). You can evaluate the model by passing the trained model file as an argument:

#### Arguments

- `model_file` (str): Path to the saved model file to evaluate.

#### Example

```bash
python evaluate.py model.joblib
```

This command will load the model from `model.joblib`, evaluate it on the `eval1.joblib` dataset, and print the evaluation score.

**Note:** The `eval1.joblib` dataset must be present in the `data/` directory for evaluation to work.

## System Overview

1. **Input Processing:** Each image pair is reshaped into grayscale images of size 62x47 pixels and concatenated into a single image of size 62x94 for further processing.
2. **Data Augmentation:** Augmentations such as horizontal flips, brightness/contrast adjustments, and Gaussian blur are applied. The optimal augmentation ratio is 0.5.
3. **Dimensionality Reduction:** PCA is applied to reduce the input data dimensionality, with 70 components providing the best results.
4. **Model Stacking:** The system uses a stacking classifier consisting of:
   - SVC (Support Vector Classifier)
   - MLP (Multi-Layer Perceptron)
   - GBC (Gradient Boosting Classifier)
   The outputs from the base models are combined using a Logistic Regression meta-classifier.

## Results

- **Accuracy:** The final system achieved an accuracy of **69.8%** on the evaluation dataset, outperforming the baseline model's accuracy of **56.3%**.
- **Confusion Matrix:** The model correctly identified 357 same-class pairs and 341 different-class pairs, with errors in subtle image differences or occlusions.
