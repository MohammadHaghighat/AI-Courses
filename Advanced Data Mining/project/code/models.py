# === File: models.py ===
import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_backbone():
    """
    Returns a ResNet-18 model with a modified final layer for 4-class rotation prediction.
    """
    # Load a pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features
    
    # Replace the final fully connected layer
    # The output is 4 for the 4 rotation angles (0, 90, 180, 270)
    model.fc = nn.Linear(num_ftrs, 4)
    
    return model

def get_mlp_backbone(input_dim, output_dim, hidden_dim=64):
    """
    Returns a simple MLP for tabular data.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, output_dim)
    )