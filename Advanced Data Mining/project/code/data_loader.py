# === File: data_loader.py (Complete and Final Version) ===
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile
import io

# ======================================================================
#  Function for CIFAR-10 (This was likely deleted by mistake)
# ======================================================================
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

# ======================================================================
#  Function for Thyroid (This is the version that works)
# ======================================================================
def get_thyroid_dataloaders(anomaly_ratio, test_size=0.5, random_state=42):
    """
    Handles downloading the standard 'annthyroid' dataset version and preparing it for the SRR framework.
    This version uses the 'ann-train.data' and 'ann-test.data' files.
    """
    url = "https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip"
    combined_file_path = "./data/thyroid_ann_combined.csv"

    if not os.path.exists(combined_file_path):
        print(f"Downloading and combining standard Thyroid 'ann' dataset files...")
        os.makedirs(os.path.dirname(combined_file_path), exist_ok=True)
        
        response = requests.get(url)
        response.raise_for_status()
        
        df_list = []
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            files_to_load = ['ann-train.data', 'ann-test.data']
            for file_name in files_to_load:
                with z.open(file_name) as f:
                    df_part = pd.read_csv(io.TextIOWrapper(f), header=None, sep=' ', usecols=range(22))
                    df_list.append(df_part)

        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(combined_file_path, index=False)
        print("Combining and saving complete.")

    print("Loading and preprocessing combined Thyroid 'ann' data...")
    df = pd.read_csv(combined_file_path)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    y_binary = np.where(y == 3, 1, 0)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_normal = X_scaled[y_binary == 0]
    X_anomaly = X_scaled[y_binary == 1]
    
    X_normal_train_pool, X_normal_test = train_test_split(
        X_normal, test_size=test_size, random_state=random_state
    )
    
    num_anomalies_to_inject = int(len(X_normal_train_pool) * anomaly_ratio / (1 - anomaly_ratio)) if anomaly_ratio > 0 else 0
    if num_anomalies_to_inject > len(X_anomaly):
        num_anomalies_to_inject = len(X_anomaly)
    
    if num_anomalies_to_inject > 0:
        anomaly_sample_indices = np.random.choice(len(X_anomaly), num_anomalies_to_inject, replace=False)
        anomalies_for_train = X_anomaly[anomaly_sample_indices]
        unlabeled_train_data = np.vstack([X_normal_train_pool, anomalies_for_train])
    else:
        unlabeled_train_data = X_normal_train_pool

    test_data = np.vstack([X_normal_test, X_anomaly])
    test_labels = np.array([0] * len(X_normal_test) + [1] * len(X_anomaly))
    
    print(f"Thyroid 'ann' data prepared. Training set size: {len(unlabeled_train_data)}, Test set size: {len(test_data)}")
    
    return unlabeled_train_data, test_data, test_labels
