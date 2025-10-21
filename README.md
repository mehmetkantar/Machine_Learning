# Naive Bayes Classifier for Bird Sound Classification

A machine learning project that implements a Naive Bayes classifier from scratch to classify bird sounds based on audio features.

## Overview

This project uses a Gaussian Naive Bayes classifier to identify three bird species from their sound recordings:
- **Major** (Great Tit)
- **Merula** (Common Blackbird)
- **Palumbus** (Wood Pigeon)

## Dataset

The dataset consists of audio features extracted from bird sound recordings:
- Training set: 300 samples
- Test set: 100 samples
- Features: Multiple audio characteristics per recording
- Classes: 3 bird species

## Implementation

The classifier is implemented from scratch in Python using NumPy, including:

### Key Functions

1. **`estimate_prior_probabilities(y)`**
   - Calculates class prior probabilities from training labels
   - Returns: Array of shape (K,) where K is the number of classes

2. **`estimate_means_and_deviations(X, y)`**
   - Estimates mean and standard deviation for each feature per class
   - Assumes Gaussian distribution for features
   - Returns: Two arrays of shape (K, D) for means and deviations

3. **`calculate_score_values(X, means, deviations, class_priors)`**
   - Computes log-likelihood scores for classification
   - Uses Gaussian probability density function
   - Returns: Score matrix of shape (N, K)

4. **`calculate_confusion_matrix(y_truth, scores)`**
   - Generates confusion matrix from predictions
   - Returns: Confusion matrix of shape (K, K)

## Files

- `0080393.py` - Main implementation of the Naive Bayes classifier
- `play_bird_sounds.ipynb` - Jupyter notebook for playing and visualizing bird sounds
- `bird_sounds_features_train.csv` - Training features
- `bird_sounds_features_test.csv` - Test features
- `bird_sounds_labels_train.csv` - Training labels
- `bird_sounds_labels_test.csv` - Test labels
- `bird_sounds_files_train.csv` - Training audio file references
- `bird_sounds_files_test.csv` - Test audio file references
- `*.wav` - Sample bird sound recordings
- `*.jpg` - Bird images

## Usage

### Running the Classifier

```bash
python 0080393.py
```

This will:
1. Load training and test data
2. Estimate class priors and feature distributions
3. Calculate classification scores
4. Generate confusion matrices
5. Print training and test accuracies

### Exploring Bird Sounds

Open the Jupyter notebook to listen to bird sounds and view images:

```bash
jupyter notebook play_bird_sounds.ipynb
```

## Results

The classifier outputs:
- Confusion matrices for both training and test sets
- Classification accuracy percentages
- Score values for each data point

## Requirements

```
numpy
pandas
matplotlib
jupyter (for notebook)
IPython (for audio playback)
```

## Installation

```bash
pip install numpy pandas matplotlib jupyter
```

## Author

Student ID: 0080393

## License

This project is part of a machine learning course assignment.
