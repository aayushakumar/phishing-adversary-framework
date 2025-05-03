## Analysis & Evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import os
import datetime
import numpy as np    
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def plot_metrics_over_rounds(controller):
    """Plot metrics evolution over rounds
    
    Args:
        controller: AdversarialController instance
    """
    if not controller.round_history:
        print("No rounds to plot")
        return
    
    # Extract data
    rounds = [r['round'] for r in controller.round_history]
    detection_rates = [r['metrics']['detection_rate'] for r in controller.round_history]
    evasion_rates = [r['metrics']['evasion_rate'] for r in controller.round_history]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, detection_rates, 'b-o', label='Detection Rate')
    plt.plot(rounds, evasion_rates, 'r-o', label='Evasion Rate')
    plt.xlabel('Round')
    plt.ylabel('Rate')
    plt.title('Detection and Evasion Rates over Rounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rounds)
    plt.ylim(0, 1)
    
    # Add confidence intervals (simplified)
    for i, r in enumerate(rounds):
        # Assuming a binomial distribution for success/failure
        n = controller.round_history[i]['attack_size']
        p_detect = detection_rates[i]
        std_dev = np.sqrt((p_detect * (1 - p_detect)) / n)
        ci = 1.96 * std_dev  # 95% confidence interval
        
        plt.errorbar(r, p_detect, yerr=ci, fmt='none', ecolor='blue', capsize=5, alpha=0.5)
    
    plt.savefig('results/metrics_evolution.png')
    plt.close()
    
    # Plot strategy-specific metrics if available
    strategies = set()
    for round_result in controller.round_history:
        if 'strategy_metrics' in round_result['metrics']:
            strategies.update(round_result['metrics']['strategy_metrics'].keys())
    
    if strategies:
        plt.figure(figsize=(12, 8))
        
        for strategy in sorted(strategies):
            strategy_evasion = []
            
            for r in controller.round_history:
                metrics = r['metrics']
                if ('strategy_metrics' in metrics and 
                    strategy in metrics['strategy_metrics'] and
                    'evasion_rate' in metrics['strategy_metrics'][strategy]):
                    strategy_evasion.append(metrics['strategy_metrics'][strategy]['evasion_rate'])
                else:
                    strategy_evasion.append(None)  # Missing data point
            
            # Plot only if we have data
            valid_points = [(r, e) for r, e in zip(rounds, strategy_evasion) if e is not None]
            if valid_points:
                x, y = zip(*valid_points)
                plt.plot(x, y, 'o-', label=f'Strategy: {strategy}')
        
        plt.xlabel('Round')
        plt.ylabel('Evasion Rate')
        plt.title('Evasion Rate by Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.savefig('results/strategy_performance.png')
        plt.close()


def feature_importance_analysis(defender, test_df):
    """Analyze feature importance
    
    Args:
        defender: Defender instance (must have handcrafted features)
        test_df: Test dataframe
    """
    
    # Check if defender has handcrafted features
    if not hasattr(defender, 'handcrafted_features'):
        print("Defender doesn't have handcrafted features for analysis")
        return
    
    # Extract features
    X = defender.handcrafted_features.extract_features(test_df['text'].tolist())
    y = test_df['label'].values
    
    # Train a simple random forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature names
    feature_names = defender.handcrafted_features.get_feature_names()
    
    # Calculate feature importances
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    
    # Permutation importance (alternative measure)
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    
    # Sort features by permutation importance
    perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
    
    # Plot permutation importances
    plt.figure(figsize=(12, 8))
    plt.title('Permutation Feature Importances')
    plt.bar(range(X.shape[1]), 
            perm_importance.importances_mean[perm_indices],
            yerr=perm_importance.importances_std[perm_indices],
            align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in perm_indices], rotation=45)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('results/permutation_importance.png')
    plt.close()


def statistical_analysis(controller, test_metrics):
    """Perform statistical analysis on results
    
    Args:
        controller: AdversarialController instance
        test_metrics: Test evaluation metrics
    """
    import scipy.stats as stats
    
    # Prepare report
    report = []
    report.append("# Statistical Analysis Report\n")
    report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Basic test metrics
    report.append("## Test Set Performance\n")
    report.append(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
    report.append(f"Precision: {test_metrics['precision']:.4f}\n")
    report.append(f"Recall: {test_metrics['recall']:.4f}\n")
    report.append(f"F1 Score: {test_metrics['f1']:.4f}\n")
    report.append(f"ROC AUC: {test_metrics['roc_auc']:.4f}\n\n")
    
    # 2. Confusion matrix
    if 'confusion_matrix' in test_metrics:
        cm = np.array(test_metrics['confusion_matrix'])
        report.append("## Confusion Matrix\n")
        report.append("\n")
        report.append("Predicted →  | Legitimate | Phishing\n")
        report.append("Actual ↓     |            |\n")
        report.append(f"Legitimate   | {cm[0][0]:9d} | {cm[0][1]:9d}\n")
        report.append(f"Phishing     | {cm[1][0]:9d} | {cm[1][1]:9d}\n")
        report.append("\n\n")
    
    # 3. Trend analysis over rounds
    if len(controller.round_history) > 1:
        report.append("## Trend Analysis\n")
        
        # Extract detection rates
        rounds = [r['round'] for r in controller.round_history]
        detection_rates = [r['metrics']['detection_rate'] for r in controller.round_history]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(rounds, detection_rates)
        
        report.append("### Detection Rate Trend\n")
        report.append(f"Slope: {slope:.4f} (change in detection rate per round)\n")
        report.append(f"R-squared: {r_value**2:.4f} (goodness of fit)\n")
        report.append(f"P-value: {p_value:.4f} (significance)\n")
        
        if p_value < 0.05:
            if slope > 0:
                report.append("Conclusion: Detection rate is **significantly increasing** over rounds.\n\n")
            else:
                report.append("Conclusion: Detection rate is **significantly decreasing** over rounds.\n\n")
        else:
            report.append("Conclusion: No significant trend in detection rate over rounds.\n\n")
    
    # 4. Strategy comparison (if multiple strategies were used)
    strategies = {}
    for round_result in controller.round_history:
        strategy = round_result['strategy']
        if strategy not in strategies:
            strategies[strategy] = []
        strategies[strategy].append(round_result['metrics']['detection_rate'])
    
    if len(strategies) > 1:
        report.append("## Strategy Comparison\n")
        
        for strategy, rates in strategies.items():
            avg_rate = np.mean(rates)
            std_rate = np.std(rates)
            report.append(f"### Strategy: {strategy}\n")
            report.append(f"Average Detection Rate: {avg_rate:.4f}\n")
            report.append(f"Standard Deviation: {std_rate:.4f}\n")
            report.append(f"Number of Rounds: {len(rates)}\n\n")
        
        # ANOVA test if we have enough data
        if all(len(rates) >= 2 for rates in strategies.values()):
            strategy_groups = list(strategies.values())
            
            # Skip if only one strategy or all strategies have a single round
            if len(strategy_groups) > 1 and sum(len(g) > 1 for g in strategy_groups) > 1:
                f_val, p_val = stats.f_oneway(*strategy_groups)
                
                report.append("### ANOVA Test for Strategy Comparison\n")
                report.append(f"F-value: {f_val:.4f}\n")
                report.append(f"P-value: {p_val:.4f}\n")
                
                if p_val < 0.05:
                    report.append("Conclusion: There is a **significant difference** between strategies.\n\n")
                else:
                    report.append("Conclusion: No significant difference detected between strategies.\n\n")
    
    # Save the report
    os.makedirs('results', exist_ok=True)
    with open('results/statistical_analysis.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("Statistical analysis completed and saved to results/statistical_analysis.md")


def generate_report(controller, test_metrics):
    """Generate a comprehensive report
    
    Args:
        controller: AdversarialController instance
        test_metrics: Test evaluation metrics
    """
    # Create report
    report = []
    report.append("# Adversarial Phishing Detection Report\n")
    report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Overview
    report.append("## Overview\n")
    report.append("This report summarizes the performance of an adversarial phishing detection system ")
    report.append("that uses a game-theoretic approach for generating and defending against ")
    report.append("phishing attacks. The system implements an iterative attacker-defender loop ")
    report.append("where both sides adapt to the other's behavior.\n\n")
    
    # 2. Summary of rounds
    report.append("## Adversarial Training Summary\n")
    report.append(f"Total Rounds: {len(controller.round_history)}\n")
    
    if controller.round_history:
        report.append("\n### Round Metrics\n")
        report.append("| Round | Strategy | Attack Size | Detection Rate | Evasion Rate |\n")
        report.append("|-------|----------|-------------|----------------|-------------|\n")
        
        for round_result in controller.round_history:
            r = round_result['round']
            strategy = round_result['strategy']
            attack_size = round_result['attack_size']
            detection = round_result['metrics']['detection_rate']
            evasion = round_result['metrics']['evasion_rate']
            
            report.append(f"| {r} | {strategy} | {attack_size} | {detection:.4f} | {evasion:.4f} |\n")
        
        report.append("\n")
    
    # 3. Final evaluation
    report.append("## Final Evaluation\n")
    report.append("The system was evaluated on a test set of emails not used during training.\n\n")
    
    report.append("### Test Metrics\n")
    report.append(f"- Accuracy: {test_metrics['accuracy']:.4f}\n")
    report.append(f"- Precision: {test_metrics['precision']:.4f}\n")
    report.append(f"- Recall: {test_metrics['recall']:.4f}\n")
    report.append(f"- F1 Score: {test_metrics['f1']:.4f}\n")
    report.append(f"- ROC AUC: {test_metrics['roc_auc']:.4f}\n\n")
    
    # 4. Visualizations
    report.append("## Visualizations\n")
    report.append("### Detection and Evasion Rates Over Rounds\n")
    report.append("![Metrics Evolution](metrics_evolution.png)\n\n")
    
    report.append("### Strategy Performance\n")
    report.append("![Strategy Performance](strategy_performance.png)\n\n")
    
    report.append("### Feature Importance\n")
    report.append("![Feature Importance](feature_importance.png)\n\n")
    
    # 5. Conclusions
    report.append("## Conclusions\n")
    
    # Basic conclusions based on metrics
    if controller.round_history:
        first_round = controller.round_history[0]['metrics']['detection_rate']
        last_round = controller.round_history[-1]['metrics']['detection_rate']
        
        if last_round > first_round:
            diff = last_round - first_round
            report.append(f"The defender improved over time, increasing detection rate by {diff:.2%}. ")
            report.append("This suggests that the adaptive learning approach was effective.\n\n")
        elif last_round < first_round:
            diff = first_round - last_round
            report.append(f"The defender's performance decreased by {diff:.2%} over time. ")
            report.append("This suggests that the attacker's adaptations were outpacing the defender's learning.\n\n")
        else:
            report.append("The defender's performance remained stable over time.\n\n")
    
    # Final test performance assessment
    f1 = test_metrics['f1']
    if f1 > 0.9:
        report.append("The final model demonstrates excellent performance on the test set, ")
        report.append("with high precision and recall, indicating effective detection of phishing emails.\n\n")
    elif f1 > 0.8:
        report.append("The final model shows good performance on the test set, ")
        report.append("with a balance of precision and recall that would be suitable for real-world use.\n\n")
    elif f1 > 0.7:
        report.append("The final model demonstrates moderate performance that could be improved further. ")
        report.append("Additional training or feature engineering may be beneficial.\n\n")
    else:
        report.append("The final model's performance on the test set is below expectations. ")
        report.append("Significant improvements are needed before deployment in a real environment.\n\n")
    
    # Save the report
    os.makedirs('results', exist_ok=True)
    with open('results/final_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("Final report generated and saved to results/final_report.md")
