# === File: data_loader.py (Using 'requests' library) ===
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
import os
import requests # Use the requests library instead of urllib

def get_cifar10_dataloaders(normal_class, anomaly_ratio, batch_size=128):
    """
    Prepares CIFAR-10 training and testing dataloaders for anomaly detection.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    normal_indices = [i for i, target in enumerate(train_dataset.targets) if target == normal_class]
    normal_data = [train_dataset[i][0] for i in normal_indices]
    anomaly_candidates_indices = [i for i, target in enumerate(train_dataset.targets) if target != normal_class]
    np.random.shuffle(anomaly_candidates_indices)
    num_anomalies_to_inject = int(len(normal_data) * anomaly_ratio / (1 - anomaly_ratio)) if anomaly_ratio > 0 else 0
    anomaly_indices_to_inject = anomaly_candidates_indices[:num_anomalies_to_inject]
    anomaly_data = [train_dataset[i][0] for i in anomaly_indices_to_inject]
    unlabeled_train_data = normal_data + anomaly_data
    shuffled_indices = np.random.permutation(len(unlabeled_train_data))
    unlabeled_train_data = [unlabeled_train_data[i] for i in shuffled_indices]
    test_normal_indices = [i for i, target in enumerate(test_dataset.targets) if target == normal_class]
    test_anomaly_indices = [i for i, target in enumerate(test_dataset.targets) if target != normal_class]
    test_normal_data = [test_dataset[i][0] for i in test_normal_indices]
    test_anomaly_data = [test_dataset[i][0] for i in test_anomaly_indices[:len(test_normal_data)]]
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

    # --- THE FIX IS HERE: Using the 'requests' library ---
    if not os.path.exists(file_path):
        print("Downloading Annthyroid dataset...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            # Use requests.get which is often more robust
            response = requests.get(url, headers=headers, allow_redirects=True, timeout=15)
            response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download the dataset. Error: {e}")
            print("Please try downloading the file manually from the URL and placing it in the './data/' directory.")
            raise

    data_mat = loadmat(file_path)
    X = data_mat['X']
    y = data_mat['y'].ravel()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    normal_data = X[y == 0]
    anomaly_data = X[y == 1]
    np.random.shuffle(normal_data)
    split_idx = int(len(normal_data) * (1 - test_size))
    normal_train_pool = normal_data[:split_idx]
    normal_test_pool = normal_data[split_idx:]
    num_anomalies_to_inject = int(len(normal_train_pool) * anomaly_ratio / (1 - anomaly_ratio)) if anomaly_ratio > 0 else 0
    np.random.shuffle(anomaly_data)
    anomalies_to_inject = anomaly_data[:num_anomalies_to_inject]
    unlabeled_train_data = np.vstack([normal_train_pool, anomalies_to_inject])
    remaining_anomalies = anomaly_data[num_anomalies_to_inject:]
    test_data = np.vstack([normal_test_pool, remaining_anomalies])
    test_labels = np.array([0] * len(normal_test_pool) + [1] * len(remaining_anomalies))
    return unlabeled_train_data, test_data, test_labels
