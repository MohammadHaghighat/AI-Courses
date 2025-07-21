# === File: main.py ===
import argparse
from data_loader import get_cifar10_dataloaders, get_thyroid_dataloaders
from models import get_resnet18_backbone, get_mlp_backbone
from trainer import SRRTrainer
from evaluation import plot_results
import numpy as np

def main(args):
    # --- Configuration ---
    config = {
        'dataset': args.dataset,
        'lr': 1e-4,
        'batch_size': 128,
        'srr_iterations': 12,
        'refinement_iters': [1, 2, 5, 10], # Iterations at which to refine data
        'ssl_epochs_per_iter': 5,
        'anomaly_ratio': args.anomaly_ratio
    }
    
    # --- Data Loading ---
    print(f"Loading dataset: {config['dataset']} with anomaly ratio: {config['anomaly_ratio']:.2f}")
    if config['dataset'] == 'cifar10':
        # CIFAR-10 class names: 0:plane, 1:car, 2:bird, 3:cat, 4:deer, 5:dog, 6:frog, 7:horse, 8:ship, 9:truck
        unlabeled_data, test_data, test_labels = get_cifar10_dataloaders(
            normal_class=args.normal_class, 
            anomaly_ratio=config['anomaly_ratio'],
            batch_size=config['batch_size']
        )
        model = get_resnet18_backbone()
    elif config['dataset'] == 'thyroid':
        unlabeled_data, test_data, test_labels = get_thyroid_dataloaders(
            anomaly_ratio=config['anomaly_ratio']
        )
        input_dim = unlabeled_data.shape[1]
        model = get_mlp_backbone(input_dim=input_dim, output_dim=10) # output_dim for tabular is arbitrary for this simplified SSL
    else:
        raise ValueError("Invalid dataset specified. Choose 'cifar10' or 'thyroid'.")

    # --- Training ---
    trainer = SRRTrainer(model, unlabeled_data, test_data, test_labels, config)
    result = trainer.run()
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SRR Framework Implementation")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'thyroid'],
                        help="Dataset to use for the experiment.")
    parser.add_argument('--normal_class', type=int, default=3,
                        help="The class to be treated as 'normal' for CIFAR-10 (0-9). Default is 3 (cat).")
    
    # To run a full analysis as per the project description, you'd loop over this
    # Example: python main.py --dataset cifar10
    # This will run for a single ratio.
    
    # --- Full Analysis Loop ---
    # To generate the plots required by the project, you should run the main function in a loop.
    run_full_analysis = True # Set to False to run a single experiment
    args = parser.parse_args()
    
    if run_full_analysis:
        anomaly_ratios = [0, 0.01, 0.02, 0.05, 0.10]
        results = []
        
        for ratio in anomaly_ratios:
            args.anomaly_ratio = ratio
            result = main(args)
            if args.dataset == 'cifar10':
                results.append(result[0]) # Store AUC
            else:
                results.append(result) # Store F1
        
        print("\n--- Full Analysis Results ---")
        for ratio, score in zip(anomaly_ratios, results):
            print(f"Anomaly Ratio: {ratio:.2f} -> Performance: {score:.4f}")
            
        metric = "AUC" if args.dataset == 'cifar10' else "F1-Score"
        plot_results(results, metric, anomaly_ratios)

    else:
        # Run a single experiment
        parser.add_argument('--anomaly_ratio', type=float, default=0.05,
                            help="The ratio of anomalies to inject into the training data.")
        args = parser.parse_args()
        main(args)