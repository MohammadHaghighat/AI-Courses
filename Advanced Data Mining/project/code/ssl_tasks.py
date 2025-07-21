# === File: ssl_tasks.py ===
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate

class RotationPredictionDataset(Dataset):
    """
    A PyTorch Dataset for the Rotation Prediction self-supervised task.
    """
    def __init__(self, image_data):
        self.image_data = image_data
        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        
        # Pick a random rotation angle
        rotation_label = torch.randint(0, 4, (1,)).item()
        angle = self.angles[rotation_label]
        
        # Apply rotation
        rotated_image = rotate(image, angle)
        
        return rotated_image, rotation_label

# Note: The GOAD-style transformation for tabular data is simpler to implement
# directly inside the training loop, as it doesn't require a custom Dataset class.
# We will apply transformations on-the-fly to batches.