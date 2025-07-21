# === File: trainer.py ===
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
        # Initial set of indices is all data
        refined_indices = np.arange(len(self.unlabeled_data))
        
        # SRR Iterative Loop
        for srr_iter in range(1, self.config['srr_iterations'] + 1):
            print(f"\n--- SRR Iteration {srr_iter}/{self.config['srr_iterations']} ---")
            
            # 1. Intermittent Data Refinement
            # As per paper, update OCCs at specific iterations
            if srr_iter in self.config['refinement_iters']:
                refined_indices = self.refinement_module.refine(
                    self.model, self.unlabeled_data, self.config['anomaly_ratio'], self.device, self.is_tabular
                )

            # 2. Update Learner on Refined Data
            if self.is_tabular:
                # For tabular, we create a simple tensor dataset
                refined_subset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(self.unlabeled_data[refined_indices])
                )
                # Note: SSL task for tabular would be more complex, for now we just train on clean data
                # A proper GOAD implementation would be needed here.
                # For simplicity, this example will proceed without the transformation SSL task for tabular.
                print("Tabular SSL task is simplified in this example. Training on refined data.")

            else: # Image data
                refined_subset = Subset(self.unlabeled_data, refined_indices)
                ssl_dataset = RotationPredictionDataset(refined_subset)
                
            ssl_loader = DataLoader(ssl_dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            # Train for a few epochs on the SSL task
            for epoch in range(self.config['ssl_epochs_per_iter']):
                loss = self.train_ssl_epoch(ssl_loader)
                print(f"SSL Epoch {epoch+1}/{self.config['ssl_epochs_per_iter']}, Loss: {loss:.4f}")

        # --- Final Evaluation ---
        print("\n--- Training complete. Starting final evaluation. ---")
        
        # 1. Perform one final refinement
        final_refined_indices = self.refinement_module.refine(
            self.model, self.unlabeled_data, self.config['anomaly_ratio'], self.device, self.is_tabular
        )
        
        # 2. Get representations of the final refined dataset
        if self.is_tabular:
            final_refined_data = self.unlabeled_data[final_refined_indices]
        else:
            final_refined_data = Subset(self.unlabeled_data, final_refined_indices)

        final_reps = self.refinement_module.get_representations(self.model, final_refined_data, self.device, self.is_tabular)
        
        # 3. Train a final GDE-OCC for evaluation
        final_occ = GaussianMixture(n_components=1, covariance_type='full')
        final_occ.fit(final_reps)
        
        # 4. Evaluate on the test set
        test_reps = self.refinement_module.get_representations(self.model, self.test_data, self.device, self.is_tabular)
        anomaly_scores = -final_occ.score_samples(test_reps) # Higher score = more anomalous
        
        if self.is_tabular:
            f1 = calculate_f1(self.test_labels, anomaly_scores)
            print(f"Final F1-Score: {f1:.4f}")
            return f1
        else:
            auc, ap = calculate_auc_ap(self.test_labels, anomaly_scores)
            print(f"Final AUC: {auc:.4f}, Final AP: {ap:.4f}")
            return auc, ap