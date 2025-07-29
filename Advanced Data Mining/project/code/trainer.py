# === File: trainer.py (Final Corrected Version) ===
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from ssl_tasks import RotationPredictionDataset
from refinement import DataRefinementModule
from evaluation import calculate_auc_ap, calculate_f1
import numpy as np
from sklearn.mixture import GaussianMixture

class SRRTrainer:
    def __init__(self, model, unlabeled_data, test_data, test_labels, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.unlabeled_data = unlabeled_data
        self.test_data = test_data
        self.test_labels = test_labels
        
        self.config = config
        self.is_tabular = config['dataset'] == 'thyroid'
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.refinement_module = DataRefinementModule(K=5)

    def train_ssl_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def run(self):
        refined_indices = np.arange(len(self.unlabeled_data))
        
        for srr_iter in range(1, self.config['srr_iterations'] + 1):
            print(f"\n--- SRR Iteration {srr_iter}/{self.config['srr_iterations']} ---")
            
            if srr_iter in self.config['refinement_iters']:
                refined_indices = self.refinement_module.refine(
                    self.model, self.unlabeled_data, self.config['anomaly_ratio'], self.device, self.is_tabular
                )

            # ========================= THE FIX IS HERE =========================
            if self.is_tabular:
                print("Creating a simplified SSL task for tabular data...")
                refined_data = self.unlabeled_data[refined_indices]
                # Create random dummy labels for the SSL pretext task.
                # The number of classes (10) must match the MLP's output_dim.
                dummy_targets = torch.randint(0, 10, (len(refined_data),))
                # Create a TensorDataset that provides both data and targets.
                # This defines 'ssl_dataset' for the tabular case.
                ssl_dataset = TensorDataset(
                    torch.FloatTensor(refined_data),
                    dummy_targets
                )
            else: # Image data
                # For images, we use a subset of the original data list
                refined_image_data = [self.unlabeled_data[i] for i in refined_indices]
                ssl_dataset = RotationPredictionDataset(refined_image_data)
            # ===================================================================
            
            ssl_loader = DataLoader(ssl_dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            for epoch in range(self.config['ssl_epochs_per_iter']):
                loss = self.train_ssl_epoch(ssl_loader)
                print(f"SSL Epoch {epoch+1}/{self.config['ssl_epochs_per_iter']}, Loss: {loss:.4f}")

        # --- Final Evaluation ---
        print("\n--- Training complete. Starting final evaluation. ---")
        
        final_refined_indices = self.refinement_module.refine(
            self.model, self.unlabeled_data, self.config['anomaly_ratio'], self.device, self.is_tabular
        )
        
        if self.is_tabular:
            final_refined_data = self.unlabeled_data[final_refined_indices]
        else:
            # Correctly create a list for image data
            final_refined_data = [self.unlabeled_data[i] for i in final_refined_indices]

        final_reps = self.refinement_module.get_representations(self.model, final_refined_data, self.device, self.is_tabular)
        
        final_occ = GaussianMixture(n_components=1, covariance_type='full')
        final_occ.fit(final_reps)
        
        test_reps = self.refinement_module.get_representations(self.model, self.test_data, self.device, self.is_tabular)
        anomaly_scores = -final_occ.score_samples(test_reps)
        
        if self.is_tabular:
            f1 = calculate_f1(self.test_labels, anomaly_scores)
            print(f"Final F1-Score: {f1:.4f}")
            return f1
        else:
            auc, ap = calculate_auc_ap(self.test_labels, anomaly_scores)
            print(f"Final AUC: {auc:.4f}, Final AP: {ap:.4f}")
            return auc, ap
