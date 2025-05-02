# phishing-adversary-framework

*A robust adversarial training loop for phishing email detection using reinforcement learning.*

---

## Overview

This repository contains the code, data, and experiment scripts for **phishing-adversary-framework**, an end-to-end framework that alternates between:

1. **Attacker**: A reinforcement learning (RL) agent that crafts minimally invasive edits to phishing emails to evade detection.
2. **Defender**: A BERT-based classifier retrained on both clean and adversarial examples to improve robustness.

By iteratively training on adversarial samples, the framework significantly boosts detection rates under attack while maintaining high accuracy on benign emails.

## Features

* **RL-Based Attacker**: Policy-gradient agent for text edits (synonym swaps, header tricks, paraphrasing).
* **Adversarial Loop**: Alternating attacker/defender updates until robust performance converges.
* **Calibration**: Temperature scaling to reduce expected calibration error (ECE).
* **Reproducibility**: Dockerfile, requirements.txt, and seed-logging for identical experiment reruns.
* **Dataset**: Combined real (3,500) and synthetic (1,500) phishing emails plus 5,000 legitimate (ham) emails.

## Getting Started

### Prerequisites

* Python 3.10+
* Docker (optional, for containerized setup)
* NVIDIA GPU with CUDA (recommended for training efficiency)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/phishing-adversary-framework.git
cd phishing-adversary-framework

# (Optional) Build and run Docker container
docker build -t phish-adv .
docker run --gpus all -it --rm -v "$PWD":/workspace phish-adv

# Or install dependencies locally
pip install -r requirements.txt
```

## Usage

1. **Prepare Data**

   ```bash
   python scripts/prepare_dataset.py --input-dir data/raw --output-dir data/processed
   ```
2. **Train Baseline Classifier**

   ```bash
   python train_classifier.py --config configs/baseline.yaml
   ```
3. **Run Adversarial Loop**

   ```bash
   python run_adversarial_loop.py --config configs/adv_loop.yaml
   ```
4. **Evaluate & Calibrate**

   ```bash
   python evaluate.py --model-path checkpoints/adv_model.pt
   python calibrate.py  --model-path checkpoints/adv_model.pt --temp-output reports/temperature_analysis.png
   ```

## Repository Structure

```
├── README.md
├── Dockerfile
├── requirements.txt
├── data/
│   ├── raw/                 # Original phishing & ham emails
│   └── processed/           # Preprocessed text corpus
├── scripts/
│   ├── prepare_dataset.py   # Data parsing & splits
│   └── ...                  # Other utility scripts
├── configs/
│   ├── baseline.yaml        # Baseline training settings
│   └── adv_loop.yaml        # Adversarial loop settings
├── models/
│   ├── classifier/          # BERT fine-tune code
│   └── attacker/            # RL policy network code
├── experiments/
│   ├── logs/                # Training & evaluation logs
│   └── figures/             # Plots: robustness, calibration
├── notebooks/               # Jupyter demos & analysis
└── LICENSE
```

## Results

* **Clean Accuracy**: \~96.8% ± 0.5%
* **Robust Detection Rate**: +25% absolute improvement under attack
* **ECE Reduction**: \~70% via temperature scaling

Detailed tables and figures are available in the `experiments/figures` directory.

## Contributing

Contributions are welcome! Please:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
