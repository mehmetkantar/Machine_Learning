# Bird Sound Classification using Discrimination by Regression

This project implements a multiclass classification model using logistic regression with gradient descent to classify bird sounds based on their acoustic features.

## Overview

The implementation uses discrimination by regression with sigmoid activation functions to classify three different bird species based on their sound features. The model is trained using gradient descent optimization.

## Dataset

The project uses four CSV files:
- `bird_sounds_features_train.csv` - Training feature data
- `bird_sounds_labels_train.csv` - Training labels
- `bird_sounds_features_test.csv` - Test feature data
- `bird_sounds_labels_test.csv` - Test labels

The features are normalized using mean and standard deviation from the training set.

## Implementation Details

### Key Components

1. **Sigmoid Function**: Computes the activation scores for multiclass classification
2. **One-Hot Encoding**: Converts class labels to binary matrix representation
3. **Gradient Calculation**: Computes gradients for weight matrix W and bias w0
4. **Discrimination by Regression**: Main training loop using gradient descent

### Model Parameters

- Learning rate (η): 0.05
- Iterations: 1000
- Weight initialization: Random uniform distribution [-0.001, 0.001]

### Training Process

The model minimizes the squared error loss function:
```
Error = 0.5 * Σ(Y_truth - Y_predicted)²
```

Weights are updated using gradient descent:
```
W = W - η * ∇W
w0 = w0 - η * ∇w0
```

## Results

The script outputs:
- Learned weight matrix W
- Learned bias vector w0
- Objective values for first 10 iterations
- Error vs iteration plot (saved as `hw02_iterations.pdf`)
- Predicted class labels for train and test sets
- Confusion matrices for both train and test sets
- Training and test accuracy percentages

## Usage

Run the script:
```bash
python 0080393.py
```

Make sure all CSV data files are in the same directory.

## Dependencies

- numpy
- pandas
- matplotlib
