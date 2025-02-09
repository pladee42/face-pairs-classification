# Image Augmentation and Model Training Pipeline

This repository contains code to preprocess and augment image data, train machine learning models, and evaluate their performance. The pipeline includes data transformation, augmentation, hyperparameter tuning, PCA (Principal Component Analysis), and model stacking to achieve high accuracy.

## Features

- **Image Augmentation:** Apply multiple augmentation techniques (e.g., flips, brightness/contrast adjustments, blurs, and edge enhancements) to the dataset, improving model robustness.
- **PCA Optimization:** Perform PCA to reduce dimensionality and find the best number of components for the model.
- **Model Stacking:** Use a stacking classifier with `SVC`, `MLPClassifier`, and `GradientBoostingClassifier` to combine multiple models for better performance.
- **Hyperparameter Tuning:** Use `RandomizedSearchCV` to find optimal parameters for the models.
- **Cross-validation:** Apply cross-validation to evaluate the models and prevent overfitting.
- **Model Evaluation:** Evaluate the performance of a trained model on a separate test dataset (`eval1.joblib`).

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

## Pipeline Steps

1. **Data Transformation:**
   - Reshape the input data.
   - Apply augmentation (if enabled).
   - Normalize the images and reshape them back into 2D arrays.

2. **PCA and Scaling:**
   - Apply PCA for dimensionality reduction.
   - Standardize the data using `StandardScaler`.

3. **Model Training:**
   - Train the following models using `RandomizedSearchCV`:
     - `SVC` (Support Vector Classifier)
     - `MLPClassifier` (Multilayer Perceptron)
     - `GradientBoostingClassifier`
   - Stack the best models into a final ensemble model.

4. **Model Saving:**
   - The trained model is saved as a `joblib` pipeline for future use.

## Model Evaluation

The script uses **StratifiedKFold cross-validation** to evaluate the performance of each model with different augmentation ratios and PCA components. The best augmentation ratio and PCA components are selected based on cross-validated accuracy.