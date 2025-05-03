import torch
import random
import numpy as np
import os
from src.data_pipeline import get_processed_datasets, download_preprocessed_data
from src.attack import TemplateGenerator, PerturbationEngine, AdaptiveAttacker, GameTheoreticPlanner
from src.defense import TransformerDefender, HandcraftedFeatures, EnsembleDefender, AdaptiveDefender
from src.controller import AdversarialController
from src.analysis import plot_metrics_over_rounds, feature_importance_analysis, statistical_analysis, generate_report

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

for d in ['data/raw', 'data/processed', 'data/adversarial', 'logs', 'models', 'results']:
    os.makedirs(d, exist_ok=True)

def run_complete_pipeline(use_real_data=False, n_rounds=3, samples_per_round=20):
    """Run the complete adversarial phishing detection pipeline"""
    print("\n====== Starting Adversarial Phishing Detection Pipeline ======\n")
    
    # 1. Get datasets
    print("\n----- Data Preparation -----\n")
    if use_real_data:
        train_df, val_df, test_df = get_processed_datasets()
    else:
        train_df, val_df, test_df = download_preprocessed_data()
    
    # 2. Initialize defender components
    print("\n----- Initializing Defender -----\n")
    
    # Transformer defender
    print("Training transformer defender...")
    transformer_defender = TransformerDefender()
    transformer_metrics = transformer_defender.train(train_df, val_df, epochs=2)
    print(f"Transformer defender metrics: {transformer_metrics}")
    
    # Calibrate
    calibration_metrics = transformer_defender.calibrate(val_df)
    print(f"Calibration metrics: {calibration_metrics}")
    
    # Handcrafted features
    handcrafted_features = HandcraftedFeatures()
    
    # Create ensemble
    ensemble_defender = EnsembleDefender()
    ensemble_defender.add_defender(transformer_defender, weight=1.0)
    
    # Tune ensemble weights
    print("Tuning ensemble weights...")
    tuning_metrics = ensemble_defender.tune_weights(val_df)
    print(f"Ensemble tuning metrics: {tuning_metrics}")
    
    # Create adaptive defender
    adaptive_defender = AdaptiveDefender(ensemble_defender)
    
    # 3. Initialize attacker components
    print("\n----- Initializing Attacker -----\n")
    template_generator = TemplateGenerator()
    perturbation_engine = PerturbationEngine()
    adaptive_attacker = AdaptiveAttacker(template_generator, perturbation_engine)
    game_planner = GameTheoreticPlanner()
    
    # 4. Create controller
    print("\n----- Setting Up Adversarial Loop -----\n")
    controller = AdversarialController(adaptive_attacker, adaptive_defender, game_planner)
    
    # 5. Run adversarial loop
    print("\n----- Running Adversarial Loop -----\n")
    round_history = controller.run_adversarial_loop(
        n_rounds=n_rounds,
        samples_per_round=samples_per_round,
        adapt_attacker=True,
        adapt_defender=True
    )
    
    # 6. Final evaluation
    print("\n----- Final Evaluation -----\n")
    test_metrics = controller.evaluate(test_df)
    print(f"Final test metrics: {test_metrics}")
    plot_metrics_over_rounds(controller)
    feature_importance_analysis(adaptive_defender, test_df)
    statistical_analysis(controller, test_metrics)
    generate_report(controller, test_metrics)


if __name__ == "__main__":
    run_complete_pipeline()