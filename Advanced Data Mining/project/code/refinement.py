# === File: refinement.py (Corrected) ===
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

class DataRefinementModule:
    """
    Implements the SRR data refinement strategy using an ensemble of One-Class Classifiers.
    """
    def __init__(self, K=5):
        self.K = K  # Number of One-Class Classifiers (OCCs)
        self.occs = [GaussianMixture(n_components=1, covariance_type='full') for _ in range(K)]

    @torch.no_grad()
    def get_representations(self, model, data, device, is_tabular=False):
        model.eval()
        reps = []
        
        if is_tabular:
            # For tabular data, we can process it all at once
            data_tensor = torch.FloatTensor(data).to(device)
            # We take the features before the final classification head
            if isinstance(model, torch.nn.Sequential):
                activation = {}
                def get_activation(name):
                    def hook(model, input, output):
                        activation[name] = output.detach()
                    return hook
                
                handle = model[-3].register_forward_hook(get_activation('rep'))
                model(data_tensor)
                handle.remove()
                return activation['rep'].cpu().numpy()
            else:
                 return model(data_tensor).cpu().numpy()

        # For image data, process in batches
        data_loader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False)
        for batch in data_loader:
            if isinstance(batch, list): # Dataloader for Subset returns a list
                 batch = batch[0]
            batch = batch.to(device)

            # Use a hook to get features from the layer before the final fc layer
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    # THE FIX IS HERE: Use flatten instead of squeeze
                    # This reliably converts [B, C, 1, 1] to [B, C]
                    activation[name] = torch.flatten(output, start_dim=1).detach()
                return hook

            handle = model.avgpool.register_forward_hook(get_activation('avgpool'))
            model(batch)
            handle.remove()
            reps.append(activation['avgpool'].cpu()) # No need for squeeze anymore

        return torch.cat(reps, dim=0).numpy()

    def refine(self, model, unlabeled_data, anomaly_ratio, device, is_tabular=False):
        print("Starting data refinement...")
        
        # 1. Get current data representations
        representations = self.get_representations(model, unlabeled_data, device, is_tabular)
        
        # 2. Split data and train K OCCs on disjoint subsets
        n_samples = representations.shape[0]
        indices = np.random.permutation(n_samples)
        subset_size = n_samples // self.K
        
        for i in range(self.K):
            start = i * subset_size
            end = (i + 1) * subset_size if i < self.K - 1 else n_samples
            subset_indices = indices[start:end]
            # Handle case where subset might be too small for GMM
            if len(subset_indices) > 1:
                self.occs[i].fit(representations[subset_indices])

        # 3. Get anomaly scores from all OCCs
        all_scores = np.array([occ.score_samples(representations) for occ in self.occs])
        
        # 4. Determine thresholds and perform pseudo-labeling
        gamma = 100 * anomaly_ratio * 1.5 
        thresholds = np.percentile(all_scores, max(5, gamma), axis=1)
        
        is_pseudo_normal = np.all(all_scores > thresholds[:, np.newaxis], axis=0)
        pseudo_normal_indices = np.where(is_pseudo_normal)[0]
        
        num_removed = n_samples - len(pseudo_normal_indices)
        print(f"Refinement complete. Removed {num_removed} / {n_samples} samples.")
        
        return pseudo_normal_indices
