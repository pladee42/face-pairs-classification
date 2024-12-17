from argparse import ArgumentParser
from joblib import load, dump
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image, ImageEnhance, ImageFilter
import warnings

def numpy_to_pillow(X: np.ndarray) -> list[Image.Image]:
    """
    Convert a numpy array of images to a list of Pillow images.

    Args:
        - images_array (np.ndarray): Input images, shape (N, H, W).

    Returns:
        - list[Image.Image]: List of Pillow images.
    """
    pil_images = []
    for i in range(X.shape[0]):
        img = Image.fromarray(X[i] * 255, mode='F')
        pil_images.append(img)
    return pil_images

def augment_images(pil_images: list[Image.Image], labels: np.ndarray, augmentation_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment a set of images and labels by applying multiple random transformations.

    This function randomly selects a portion of the input images (controlled by augmentation_ratio) 
    and applies 8 augmentations (flips, brightness & contrast changes, blurs, edge detection, contour filter and edge enhancements). 
    The augmented images are appended to the original dataset, increasing both the number of images and labels.

    Args:
        - pil_images (list[Image.Image]): List of Pillow images.
        - labels (np.ndarray): Labels corresponding to each input image.
        - augmentation_ratio (float): Fraction of images to augment.

    Returns:
        (np.ndarray, np.ndarray): Augmented images and labels, with images normalized 
        to [0,1] and arranged in the same format as input.
    """
    augmented_images = list(pil_images)
    augmented_labels = list(labels)

    # Determine the number of images to augment based on augmentation_ratio
    total_images = len(pil_images)
    num_augment = int(augmentation_ratio * total_images)

    for i in range(8):  # 8 augmentation types
        augment_indices = np.random.choice(total_images, size=num_augment, replace=False)
        
        for image_idx in augment_indices:
            original_image = pil_images[image_idx]
            label = labels[image_idx]
            
            # Apply specific augmentation logic
            if i == 0:  # Horizontal flip
                augmented_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif i == 1:  # Brightness adjustment
                augmented_image = ImageEnhance.Brightness(original_image.convert('L')).enhance(np.random.uniform(0.7, 1.5))
            elif i == 2:  # Contrast adjustment
                augmented_image = ImageEnhance.Contrast(original_image.convert('L')).enhance(np.random.uniform(0.5, 2))
            elif i == 3:  # Sharpness adjustment
                augmented_image = ImageEnhance.Sharpness(original_image.convert('L')).enhance(np.random.uniform(0.5, 2))
            elif i == 4:  # Edge detection
                augmented_image = original_image.convert('L').filter(ImageFilter.FIND_EDGES)
            elif i == 5:  # Gaussian blur
                augmented_image = original_image.convert('L').filter(ImageFilter.GaussianBlur(np.random.uniform(0.7, 1)))
            elif i == 6: # Edge enhancement
                augmented_image = original_image.convert('L').filter(ImageFilter.EDGE_ENHANCE_MORE)
            elif i == 7: # Contour Filter
                augmented_image = original_image.convert('L').filter(ImageFilter.CONTOUR)

            # Add augmented image and label to the dataset
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
    
    # Normalize the images and convert labels to numpy arrays
    X_aug = np.array([np.array(img) / 255 for img in augmented_images])
    y_aug = np.array(augmented_labels)
    
    return X_aug, y_aug


def transform_train(X: np.ndarray, y: np.ndarray, augmentation_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform and augment the training data.
    
    Steps:
        1. Reshape input data to (N, H, 2*W) 
        2. Convert to Pillow images.
        3. Augment images.
        4. Reshape back into 2D arrays (N, H*W).

    Args:
        - X (np.ndarray): The images data, shape.
        - y (np.ndarray): The target labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: The transformed data and labels.
    """
    X = X.reshape(-1, 2, 62, 47)
    X = np.concatenate((X[:, 0], X[:, 1]), axis=2)
    X = numpy_to_pillow(X)
    X, y = augment_images(X, y, augmentation_ratio)
    X = X.reshape(X.shape[0], -1)    
    
    return X, y

def transform_test(X: np.ndarray) -> np.ndarray:
    """
    Transform the testing data into the format expected by the model.
    (No augmentation)

    Args:
        X (np.ndarray): The images data.

    Returns:
        np.ndarray: Flattened transformed data.
    """
    X = X.reshape(-1, 2, 62, 47)
    X = np.concatenate((X[:, 0], X[:, 1]), axis=2)
    X = X.reshape(X.shape[0], -1)
    
    return X

def find_best_augmentation(X_orig: np.ndarray, y_orig: np.ndarray) -> float:
    """
    Find the best augmentation ratio that maximizes the mean cross-validated accuracy of MLP Classifier.

    Args:
        - X_orig (np.ndarray): The images data.
        - y_orig (np.ndarray): The target labels.

    Returns:
        float: The best augmentation ratio.
    """
    augmentation_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_score = 0.0
    best_ratio = augmentation_ratios[0]  # Initialize with the first ratio
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
    
    for ratio in augmentation_ratios:
        print(f'== Evaluating augmentation ratio: {ratio} ==')
        # Apply transformation and augmentation
        X_train, y_train = transform_train(X_orig, y_orig, augmentation_ratio=ratio)
        # Define the pipeline with PCA and MLPClassifier
        pipeline = Pipeline([
            ('pca', PCA(n_components=70, random_state=101)),
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(75, 50),
                                  solver='adam',
                                  max_iter=1000,
                                  activation='relu',
                                  early_stopping=True,
                                  random_state=101))
        ])
        # Perform cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        print(f'== Cross-validated accuracy for augmentation ratio {ratio}: {mean_score:.4f} ==\n')
        
        # Check if the current score is the best
        if mean_score > best_score:
            best_score = mean_score
            best_ratio = ratio
                
    print(f'\n== Best augmentation ratio: {best_ratio} with cross-validated accuracy: {best_score:.4f} ==\n')
    
    return best_ratio


def find_best_pca(X_aug: np.ndarray, y_aug: np.ndarray) -> int:
    """
    Find the best number of components for PCA that maximizes the mean cross-validated accuracy of MLP Classifier.

    Args:
        - X_aug (np.ndarray): Transformed and augmented data.
        - y_aug (np.ndarray): Labels.

    Returns:
        int: The best number of components to use.
    """
    n_components_list = [65, 70, 75, 80, 85, 90, 95, 100]
    best_score = 0.0
    best_n = 0
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
    
    for n in n_components_list:
        print(f'== Evaluating number of PCA components: {n} ==')
        # Define the pipeline with PCA and MLPClassifier
        pipeline = Pipeline([
            ('pca', PCA(n_components=n, random_state=101)),
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(75, 50),
                                  solver='adam',
                                  max_iter=1000,
                                  activation='relu',
                                  early_stopping=True,
                                  random_state=101))
        ])
        # Perform cross-validation
        scores = cross_val_score(pipeline, X_aug, y_aug, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        print(f'== Cross-validated accuracy for n_components {n}: {mean_score:.4f} ==\n')
        
        # Check if the current score is the best
        if mean_score > best_score:
            best_score = mean_score
            best_n = n
                
    print(f'\n== Best number of PCA components: {best_n} with cross-validated accuracy: {best_score:.4f} ==\n')
    
    return best_n

def train(data_file: str, model_name: str, search_best_pca: bool = False, search_best_augmentation: bool = False) -> None:
    """
    Train and save a model pipeline.

    Steps:
        1. Load and preprocess data.
        2. (Optional) Find best augmentation ratio.
        3. Transform data and apply augmentation.
        4. (Optional) Find best PCA components.
        5. Perform hyperparameter search on multiple models.
        6. Stack models into a final ensemble.
        7. Save the pipeline.

    Args:
        - data_file (str): The training data as a dictionary with keys 'data' and 'target'.
        - model_name (str): The file name to save the model pipeline.
        - search_best_pca (bool): If True, search best PCA components.
        - search_best_augmentation (bool): If True, search best augmentation ratio.

    Returns:
        None
    """
    train_data = load(open(data_file, "rb"))
    # Default parameters
    n_components = 70
    augmentation_ratio = 0.5
    
    # Search for best augmentation ratio if requested
    if search_best_augmentation:
        augmentation_ratio = find_best_augmentation(train_data['data'], train_data['target'])
        
    # Transform and augment data
    X_train, y_train = transform_train(train_data['data'], train_data['target'], augmentation_ratio)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Search for best number of components for PCA if requested
    if search_best_pca:
        n_components = find_best_pca(X_train, y_train)

    pca = PCA(n_components=n_components, random_state=101)
    X_train = pca.fit_transform(X_train)

    # Train models using RandomizedSearchCV
    models = {
        'SVC': SVC(probability=True, random_state=101),
        'MLPClassifier': MLPClassifier(early_stopping=True, random_state=101),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=101),
        }

    param_grids = {
        'SVC': {
            'C': [1, 10, 100],
            'gamma': [0.1, 0.01, 0.001],
            'kernel': ['rbf']
            },
        'MLPClassifier': {
            'hidden_layer_sizes': [(75, 50), (50, 25), (128, 84, 48)],
            'activation': ['relu', 'logistic'],
            'alpha': [1e-8, 1e-4],
            'max_iter': [1000, 2000],
            'solver': ['adam']
            },
        'GradientBoostingClassifier': {
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'n_estimators': [100, 250, 500, 750, 1000],
            'max_depth': [3, 4, 5, 6, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'loss': ['log_loss', 'exponential']
            }
        }
    
    # Define K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
    grids = {}
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Hyperparameter tuning
    for m_name, model in models.items():
        print(f'== Training {m_name} ==')
        grids[m_name] = RandomizedSearchCV(estimator=model, param_distributions=param_grids[m_name], cv=cv, scoring='accuracy', verbose=5, n_jobs=-1, refit=True)
        grids[m_name].fit(X_train, y_train)
        best_params = grids[m_name].best_params_
        
        print(f'Best parameters for {m_name}: {best_params}\n')
    
    # Stacking Ensemble
    estimators = [
        ('svc', grids['SVC'].best_estimator_),
        ('mlp', grids['MLPClassifier'].best_estimator_), 
        ('gbc', grids['GradientBoostingClassifier'].best_estimator_)
        ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1, penalty='l2', solver='saga', class_weight='balanced'),
        cv=cv,
        n_jobs=-1,
        verbose=5
        )
    print(f'== Training StackingClassifier ==\n')
    stacking_clf.fit(X_train,y_train)
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessing', FunctionTransformer(transform_test)),
        ('scaler', scaler),
        ('pca', pca),
        ('stacking', stacking_clf)
        ])
    
    # Save model
    dump(pipeline, model_name)
    print(f'\n== Model successfully saved to {model_name} ==\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("training_data", type=str, default='train.joblib',
                        help="The file containing the training data.")
    parser.add_argument("model_file_name", type=str, default='model.joblib',
                        help="The file name to save the model.")
    # Optional boolean arguments
    parser.add_argument('--test-aug', action='store_true',
                    help='If set, will test augmentation ratio.')
    parser.add_argument('--test-pca', action='store_true',
                    help='If set, will search for best PCA components.')
    
    args = parser.parse_args()
    
    train(
        data_file=args.training_data,
        model_name=args.model_file_name,
        search_best_pca=args.test_pca,
        search_best_augmentation=args.test_aug
    )
