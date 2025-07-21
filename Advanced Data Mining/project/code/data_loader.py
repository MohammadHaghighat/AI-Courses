# === File: data_loader.py ===
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
import os
import urllib.request

def get_cifar10_dataloaders(normal_class, anomaly_ratio, batch_size=128):
    """
    Prepares CIFAR-10 training and testing dataloaders for anomaly detection.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # --- Prepare Training Data ---
    train_data = []
    train_targets = []

    # Get all "normal" data
    normal_indices = [i for i, target in enumerate(train_dataset.targets) if target == normal_class]
    normal_data = [train_dataset[i][0] for i in normal_indices]
    
    # Get potential "anomalous" data (from all other classes)
    anomaly_candidates_indices = [i for i, target in enumerate(train_dataset.targets) if target != normal_class]
    np.random.shuffle(anomaly_candidates_indices)

    # Calculate number of anomalies to inject
    num_anomalies_to_inject = int(len(normal_data) * anomaly_ratio / (1 - anomaly_ratio))
    
    # Select anomaly samples
    anomaly_indices_to_inject = anomaly_candidates_indices[:num_anomalies_to_inject]
    anomaly_data = [train_dataset[i][0] for i in anomaly_indices_to_inject]

    # Combine normal and anomalous data to create the unlabeled training set
    unlabeled_train_data = normal_data + anomaly_data
    # Create labels for our reference (0=normal, 1=anomaly) - NOT for training
    unlabeled_train_labels = [0] * len(normal_data) + [1] * len(anomaly_data)
    
    # Shuffle the unlabeled training data
    shuffled_indices = np.random.permutation(len(unlabeled_train_data))
    unlabeled_train_data = [unlabeled_train_data[i] for i in shuffled_indices]
    # We keep labels for potential analysis, but the model won't see them
    unlabeled_train_labels = [unlabeled_train_labels[i] for i in shuffled_indices]


    # --- Prepare Test Data ---
    # Test set is clean: contains normal and a balanced set of anomalous samples
    test_normal_indices = [i for i, target in enumerate(test_dataset.targets) if target == normal_class]
    test_anomaly_indices = [i for i, target in enumerate(test_dataset.targets) if target != normal_class]
    
    test_normal_data = [test_dataset[i][0] for i in test_normal_indices]
    test_anomaly_data = [test_dataset[i][0] for i in test_anomaly_indices]
    
    test_data = test_normal_data + test_anomaly_data
    test_labels = [0] * len(test_normal_data) + [1] * len(test_anomaly_data)

    return unlabeled_train_data, test_data, test_labels


def get_thyroid_dataloaders(anomaly_ratio, test_size=0.5):
    """
    Prepares Thyroid (annthyroid) dataset for anomaly detection.
    """
    url = "http://odds.cs.stonybrook.edu/annthyroid-dataset/"
    file_path = "./data/annthyroid.mat"
    
    if not os.path.exists('./data'):
        os.makedirs('./data')

    if not os.path.exists(file_path):
        print("Downloading Annthyroid dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")

    data = loadmat(file_path)
    X = data['X']
    y = data['y'].ravel()

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Separate normal and anomaly data
    normal_data = X[y == 0]
    anomaly_data = X[y == 1]
    
    # Split normal data into training and testing pools
    np.random.shuffle(normal_data)
    split_idx = int(len(normal_data) * (1 - test_size))
    normal_train_pool = normal_data[:split_idx]
    normal_test_pool = normal_data[split_idx:]

    # --- Prepare Training Data ---
    # Inject anomalies into the training pool
    num_anomalies_to_inject = int(len(normal_train_pool) * anomaly_ratio / (1 - anomaly_ratio))
    np.random.shuffle(anomaly_data)
    anomalies_to_inject = anomaly_data[:num_anomalies_to_inject]
    
    unlabeled_train_data = np.vstack([normal_train_pool, anomalies_to_inject])
    unlabeled_train_labels = np.array([0] * len(normal_train_pool) + [1] * len(anomalies_to_inject))

    # Shuffle
    shuffled_indices = np.random.permutation(len(unlabeled_train_data))
    unlabeled_train_data = unlabeled_train_data[shuffled_indices]
    unlabeled_train_labels = unlabeled_train_labels[shuffled_indices]

    # --- Prepare Test Data ---
    # The rest of the data is for testing
    remaining_anomalies = anomaly_data[num_anomalies_to_inject:]
    test_data = np.vstack([normal_test_pool, remaining_anomalies])
    test_labels = np.array([0] * len(normal_test_pool) + [1] * len(remaining_anomalies))

    return unlabeled_train_data, test_data, test_labels