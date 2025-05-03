# Phishing Adversary Framework


This repository contains the code, data, and experiment scripts for **phishing-adversary-framework**, an end-to-end framework that alternates between:

1. **Attacker**: A reinforcement learning (RL) agent that crafts minimally invasive edits to phishing emails to evade detection.
2. **Defender**: A BERT-based classifier retrained on both clean and adversarial examples to improve robustness.

By iteratively training on adversarial samples, the framework significantly boosts detection rates under attack while maintaining high accuracy on benign emails.

---
An extensible adversarial phishing email detection system that implements a full attacker–defender loop. Includes:

* **Data Pipeline**: Download & preprocess real and synthetic phishing/ham datasets.
* **Attack Module**: Template-based, perturbation-based, and learning-based phishing email generators.
* **Defense Module**: Transformer-based, handcrafted-features, ensemble, and adaptive defenses.
* **Game-Theoretic Planner**: Strategy mixing via payoff matrices & Nash equilibrium solvers.
* **Adversarial Controller**: Orchestrates iterative attacker–defender rounds, logs metrics.
* **Analysis & Reporting**: Visualization of detection/evasion over rounds, feature importance, statistical analysis, and final report generation.

<!-- --- -->

## Features

* **RL-Based Attacker**: Policy-gradient agent for text edits (synonym swaps, header tricks, paraphrasing).
* **Adversarial Loop**: Alternating attacker/defender updates until robust performance converges.
* **Calibration**: Temperature scaling to reduce expected calibration error (ECE).
* **Reproducibility**: Dockerfile, requirements.txt, and seed-logging for identical experiment reruns.
* **Dataset**: Combined real (3,500) and synthetic (1,500) phishing emails plus 5,000 legitimate (ham) emails.

## Getting Started

### Prerequisites

* Python 3.10+
<!-- * Docker (optional, for containerized setup) -->
* NVIDIA GPU with CUDA (recommended for training efficiency)

<!-- ### Installation

```bash
# Clone the repository
git clone https://github.com/aayushakumar/phishing-adversary-framework.git
cd phishing-adversary-framework

 -->
## ⚙️ Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/aayushakumar/phishing-adversary-framework.git
   cd phishing-adversary-framework
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```


## 🚀 Quickstart

### 1. Prepare Data (TO BE COMPLETED)

```bash
# Download and preprocess datasets
python scripts/prepare_data.py --raw-dir data/raw --proc-dir data/processed
```

*For demo or Kaggle usage, you can instead generate synthetic data:*

```bash
python scripts/prepare_data.py --use-synthetic
```

### 2. Train Defender

```bash
python scripts/train_defender.py \
  --train-csv data/processed/train.csv \
  --val-csv   data/processed/val.csv \
  --output    models/transformer-defender
```



---

## Results

* **Clean Accuracy**: \~96.8% ± 0.5%
* **Robust Detection Rate**: +25% absolute improvement under attack
* **ECE Reduction**: \~70% via temperature scaling

## Contributing

Contributions are welcome! Please:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
