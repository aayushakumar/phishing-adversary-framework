import json
import os
from attack import AdaptiveAttacker, GameTheoreticPlanner
from defense import AdaptiveDefender, EnsembleDefender


class AdversarialController:
    """Controller for the adversarial loop"""
    
    def __init__(self, attacker=None, defender=None, planner=None):
        """Initialize controller
        
        Args:
            attacker: Attacker object
            defender: Defender object
            planner: Game-theoretic planner
        """
        self.attacker = attacker or AdaptiveAttacker()
        self.defender = defender or EnsembleDefender()
        self.planner = planner or GameTheoreticPlanner()
        
        self.round_history = []
        self.current_round = 0
    
    def run_round(self, n_samples=10, strategy=None):
        """Run a single round in the adversarial loop
        
        Args:
            n_samples: Number of attack samples to generate
            strategy: Attack strategy (or None for planner to choose)
            
        Returns:
            Round results
        """
        self.current_round += 1
        print(f"\n===== Starting Round {self.current_round} =====")
        
        # Choose attack strategy if not specified
        if strategy is None and self.planner:
            strategy = self.planner.choose_strategy()
            print(f"Planner chose strategy: {strategy}")
        
        # Generate attack emails
        attack_emails = self.attacker.generate_attack(n_samples, strategy)
        print(f"Generated {len(attack_emails)} attack emails")
        
        # Defender labels the emails
        attack_results = self.defender.predict(attack_emails)
        print(f"Defender processed the attacks")
        
        # Calculate metrics
        metrics = self._calculate_metrics(attack_results)
        print(f"Attack metrics - Evasion rate: {metrics['evasion_rate']:.2f}, "
              f"Detection rate: {metrics['detection_rate']:.2f}")
        
        # Update the game-theoretic planner
        if self.planner:
            self.planner.update_payoffs(attack_results)
            new_distribution = self.planner.solve_mixed_strategy()
            print(f"Updated strategy distribution: {new_distribution}")
        
        # Log round results
        round_results = {
            'round': self.current_round,
            'strategy': strategy,
            'attack_size': len(attack_emails),
            'metrics': metrics,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.round_history.append(round_results)
        
        # Save round results
        self._save_round_results(round_results)
        
        return round_results
    
    def adapt_and_update(self, round_results, adapt_attacker=True, adapt_defender=True):
        """Update attacker and defender based on round results
        
        Args:
            round_results: Results from the last round
            adapt_attacker: Whether to update the attacker
            adapt_defender: Whether to update the defender
            
        Returns:
            Update metrics
        """
        print(f"\n===== Adapting after Round {self.current_round} =====")
        
        update_metrics = {}
        
        # Update attacker if required
        if adapt_attacker and hasattr(self.attacker, 'update_model'):
            attack_results = round_results.get('attack_results', [])
            if not attack_results:
                # Reconstruct from metrics
                evasion_count = int(round_results['metrics']['evasion_rate'] * 
                                   round_results['attack_size'])
                
                # Simulate attack results
                attack_results = [
                    {'detected': i >= evasion_count, 
                     'email': {'strategy': round_results['strategy']}}
                    for i in range(round_results['attack_size'])
                ]
            
            self.attacker.update_model(attack_results)
            print("Attacker model updated")
            update_metrics['attacker_updated'] = True
        
        # Update defender if required
        if adapt_defender and hasattr(self.defender, 'update'):
            attack_results = round_results.get('attack_results', [])
            defender_metrics = self.defender.update(attack_results, retrain=True)
            print("Defender updated")
            update_metrics['defender_updated'] = True
            update_metrics['defender_metrics'] = defender_metrics
        
        return update_metrics
    
    def _calculate_metrics(self, attack_results):
        """Calculate metrics for attack results
        
        Args:
            attack_results: List of dictionaries with attack results
            
        Returns:
            Dictionary of metrics
        """
        # Calculate basic metrics
        total = len(attack_results)
        detected = sum(1 for result in attack_results if result['detected'])
        evaded = total - detected
        
        detection_rate = detected / total if total > 0 else 0
        evasion_rate = evaded / total if total > 0 else 0
        
        # Group by strategy if available
        strategy_metrics = {}
        for result in attack_results:
            if 'strategy' in result:
                strategy = result['strategy']
                if strategy not in strategy_metrics:
                    strategy_metrics[strategy] = {'total': 0, 'detected': 0}
                
                strategy_metrics[strategy]['total'] += 1
                if result['detected']:
                    strategy_metrics[strategy]['detected'] += 1
        
        # Calculate per-strategy metrics
        for strategy, counts in strategy_metrics.items():
            if counts['total'] > 0:
                counts['detection_rate'] = counts['detected'] / counts['total']
                counts['evasion_rate'] = 1 - counts['detection_rate']
        
        return {
            'total': total,
            'detected': detected,
            'evaded': evaded,
            'detection_rate': detection_rate,
            'evasion_rate': evasion_rate,
            'strategy_metrics': strategy_metrics
        }
    
    def _save_round_results(self, round_results):
        """Save round results to disk
        
        Args:
            round_results: Dictionary with round results
        """
        os.makedirs('results', exist_ok=True)
        
        # Save to JSON file
        filename = f"results/round_{round_results['round']}.json"
        with open(filename, 'w') as f:
            # Clone dictionary to avoid modifying the original
            results_copy = round_results.copy()
            
            # Remove attack_results as they can be large and contain objects
            if 'attack_results' in results_copy:
                del results_copy['attack_results']
            
            json.dump(results_copy, f, indent=2, default=str)
    
    def run_adversarial_loop(self, n_rounds=5, samples_per_round=10, 
                           adapt_attacker=True, adapt_defender=True):
        """Run the complete adversarial loop
        
        Args:
            n_rounds: Number of rounds to run
            samples_per_round: Number of attack samples per round
            adapt_attacker: Whether to update the attacker after each round
            adapt_defender: Whether to update the defender after each round
            
        Returns:
            Complete history of the adversarial loop
        """
        for round_idx in range(n_rounds):
            # Run a round
            round_results = self.run_round(samples_per_round)
            
            # Update attacker and defender
            if round_idx < n_rounds - 1:  # Don't update after the last round
                self.adapt_and_update(round_results, adapt_attacker, adapt_defender)
        
        return self.round_history
    
    def evaluate(self, test_df=None):
        """Evaluate current defender on test data
        
        Args:
            test_df: Test dataframe with 'text' and 'label' columns
            
        Returns:
            Evaluation metrics
        """
        if test_df is None:
            print("No test data provided for evaluation")
            return {}
        
        # Make predictions
        predictions = self.defender.predict(test_df['text'].tolist())
        pred_labels = [1 if pred['is_phishing'] else 0 for pred in predictions]
        true_labels = test_df['label'].tolist()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels),
            'roc_auc': roc_auc_score(true_labels, [pred['phishing_prob'] for pred in predictions])
        }
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
