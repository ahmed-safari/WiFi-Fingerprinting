# WiFi Fingerprinting for Device-Robust Indoor Localization


## Overview

This project investigates device-robust indoor localization using WiFi fingerprinting. We address the **device heterogeneity problem**—where models trained on one set of devices fail to generalize to new devices—using per-sample RSSI normalization combined with Random Forest classification.

### Key Results

| Metric        | Baseline | Our Approach | Improvement |
| ------------- | -------- | ------------ | ----------- |
| LODO Test F1  | 0.797    | 0.874        | +9.6%       |
| Validation F1 | 0.809    | 0.873        | +7.8%       |

## Project Structure

```
WiFi-Fingerprinting/
├── notebook.ipynb          # Complete reproducible analysis
├── report.md               # Formal report (≤8 pages)
├── presentation_slides.md  # Presentation (10 slides)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── datasets/
│   ├── trainingData.csv    # UJIIndoorLoc training set
│   └── validationData.csv  # UJIIndoorLoc validation set
└── results/
    ├── ablation_study.csv
    ├── ablation_study.png
    ├── baseline_classification.csv
    ├── baseline_regression.csv
    ├── validation_results.csv
    ├── validation_confusion_matrices.png
    ├── device_heterogeneity.png
    └── coverage_matrix.png
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ahmed-safari/WiFi-Fingerprinting.git
cd WiFi-Fingerprinting
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Analysis

```bash
# Launch Jupyter Notebook
jupyter notebook notebook.ipynb
```

Then execute all cells in order: **Cell → Run All**

### 4. View Results

- Numerical results: `results/*.csv`
- Visualizations: `results/*.png`
- Full report: `report.md`

## Reproducibility

### Random Seeds

All stochastic processes use `random_state=42`:

- Train/test splits
- Random Forest initialization
- Any sampling operations

### Environment

Tested with:

- Python 3.10.12
- See `requirements.txt` for package versions


## Dataset

We use the [UJIIndoorLoc dataset](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc) from UCI Machine Learning Repository.

| Attribute    | Training | Validation |
| ------------ | -------- | ---------- |
| Samples      | 19,937   | 1,111      |
| WAP Features | 520      | 520        |
| Buildings    | 3        | 3          |
| Floors       | 5        | 5          |
| Devices      | 16       | 11         |

**Citation:**

```
Torres-Sospedra, J., et al. (2014). UJIIndoorLoc: A new multi-building and
multi-floor database for WLAN fingerprint-based indoor localization problems.
IPIN 2014.
```

## Methodology

### The Problem

Different smartphone models measure WiFi signal strength (RSSI) differently, causing models to fail on unseen devices.

### Our Solution

1. **Per-Sample Normalization**: Transform each sample's RSSI values relative to its own min/max, removing device-specific offsets
2. **Random Forest**: Ensemble learning captures robust, device-invariant patterns

### Evaluation Protocol

- **Leave-One-Device-Out (LODO)**: Hold out all samples from one device for testing
- **Primary Metric**: Macro F1 (handles class imbalance)

## Results Summary

### Device Generalization Gap

| Split        | Floor F1     |
| ------------ | ------------ |
| Random 80/20 | 98.9%        |
| LODO         | 79.7%        |
| **Gap**      | **-19.3 pp** |

### Our Improvement

| Method                   | LODO F1   | Δ         |
| ------------------------ | --------- | --------- |
| Baseline (k-NN + Scaler) | 79.7%     | —         |
| Per-Sample Norm + RF     | **87.4%** | **+9.6%** |

## Ethical Considerations

- **Privacy**: Location data requires careful handling; we recommend differential privacy for deployment
- **Fairness**: Some device types show lower performance; monitor per-device metrics
- **Reproducibility**: All code and data provided for verification

## AI Tool Disclosure

AI-assisted tools were used for code implementation and language improvement:

- **GitHub Copilot**: Code syntax assistance and debugging
- **DeepSeek**: Research assistance and code optimization

All research ideas, hypotheses, experimental design, and analysis interpretations were independently developed by the project team. The AI tools served purely as productivity aids.
