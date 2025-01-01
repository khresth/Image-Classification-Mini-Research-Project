# Image-Classification-Mini-Research-Project
This project was developed as part of the SCC 361 Artificial Intelligence module coursework. This includes data preparation, training/testing splits, K-Nearest Neighbors with custom distance metrics, comparisons with pre-built classification models and a report.

## Prerequisites
MATLAB (R2021a or later recommended)
CIFAR-10 dataset in .mat format (e.g., cifar-10-data.mat)

## Project Structure
cifar-10-data.mat: Contains the CIFAR-10 dataset.
image.png: Displays sample images and labels from the dataset.
cw1.mat: “classes” variable, “training_index” variable, Accuracy, confusion matrix and time taken measures.

## Features
# Dataset Loading and Visualization
Loads the CIFAR-10 dataset and normalizes pixel values.
Displays a random selection of images with their corresponding labels.

# Random Class Selection
Selects three random classes using a seeded random number generator for reproducibility.
Extracts images and labels corresponding to the selected classes.

# Data Splitting
Splits data into 50% training and 50% testing subsets.
Reshapes data for use with classifiers.

# K-Nearest Neighbors (KNN)
Implements a custom KNN function.
Evaluates using both Euclidean and Cosine distance metrics.

# Support Vector Machine (SVM)
Trains a multi-class SVM using MATLAB's fitcecoc function.

# Decision Tree
Trains a decision tree using MATLAB's fitctree function.

# Performance Metrics
Computes accuracy and confusion matrices for each classifier.
Measures execution time for training and predictions.

