# EEG-Based Mental Stress Detection

**Course:** Data Science Capstone Project  
**Authors:** Yoana Agafonova and Iliana Hristova  
**Institution:** IMC University of Applied Sciences Krems  

Binary classification of mental stress states from EEG recordings using handcrafted features and classical machine learning models.


## Problem

Mental stress is typically assessed through subjective self-reports, which are prone to bias. This project builds an objective, EEG-based classifier that distinguishes stress from non-stress states using a 5-channel EMOTIV EEG device and a machine learning pipeline.


## Dataset

**Source:** [Mendeley Data – An EEG Recordings Dataset for Mental Stress Detection](https://data.mendeley.com/datasets/wnshbvdxs2/1)

The dataset contains 112 EDF files across five experimental conditions:

| Condition | Files | Class |
|---|---|---|
| Complex Mathematical Problem Solving (CMPS) | 22 | Stress |
| Horror Video Stimulation | 22 | Stress |
| Stroop Colour Word Test (SCWT) | 24 | Stress |
| Trier Mental Challenge Test (TMCT) | 24 | Stress |
| Relaxing Music | 20 | Non-stress |

EEG channels used: **AF3, T7, Pz, T8, AF4** — recorded at 128 Hz.

**Download the dataset and place it under `data/` with the original folder names:**

```
data/
  Complex Mathematical Problem solving (CMPS)/
  Horrer Video Stimulation/
  Stroop Colour Word Test(SCWT)/
  Trier Mental Challenge Test (TMCT)/
  Participants Listening to Relaxing Music/
```


## Repository Structure

```
eeg-based-mental-stress-detection-capstone-project/
├── data/                                       # EDF dataset files (not tracked in git)
├── notebooks/
│   └── EEG_Stress_Detection.ipynb              # Full analysis notebook
├── outputs/
│   ├── plots/                                  # Generated figures
│   └── results/                                # CSV results and classification report
├── report/
│   ├── main.tex                                # LaTeX source for the final report
│   └── main.pdf                                # Compiled report
├── requirements.txt                            # Python dependencies
├── EEG_Stress_Detection_Presentation.pptx      # Slide set
└── README.md
```


## Setup

**Python version:** 3.10 or higher recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries: `mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`.


## How to Run

1. Download the dataset from the Mendeley link above and place the folders under `data/`.
2. Open and run the notebook end-to-end:

```bash
jupyter notebook notebooks/EEG_Stress_Detection.ipynb
```

The notebook is organized into the following sections:

1. Imports and Setup  
2. Dataset Loading and Label Creation  
3. Exploratory Data Analysis  
4. Signal Inspection and Visualization  
5. Feature Extraction from Full Recordings (baseline)  
6. Baseline Model Training  
7. Window-Based Feature Extraction (10-second windows)  
8. Model Comparison — StratifiedKFold and GroupKFold  
9. Final Model Analysis (Gradient Boosting)  
10. Feature Importance  

All outputs are saved automatically to `outputs/plots/` and `outputs/results/`.


## Methodology Summary

- **Preprocessing:** 0.5–45 Hz bandpass filter (MNE)
- **Segmentation:** non-overlapping 10-second windows → 3507 windows total
- **Features:** 75 per window — 10 statistical + 5 frequency-band power features × 5 channels
- **Models:** Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting (scikit-learn pipelines with StandardScaler)
- **Validation:** GroupKFold (5 folds, grouped by EDF file) to prevent data leakage


## Key Results

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---|---|
| **Gradient Boosting** | **0.865** | 0.581 | **0.593** |
| KNN | 0.835 | 0.556 | 0.562 |
| SVM | 0.703 | **0.643** | 0.561 |
| Random Forest | 0.839 | 0.539 | 0.540 |
| Logistic Regression | 0.654 | 0.620 | 0.522 |

*All metrics from 5-fold GroupKFold cross-validation.*

The final Gradient Boosting model detects stress windows with recall = 0.98 but struggles with non-stress recall = 0.16 due to class imbalance (86.6% stress / 13.4% non-stress).

The most important feature is **gamma-band power at channel AF4** (26% importance), followed by theta power at AF4 and gamma power at T8.


## Limitations

- Severe class imbalance was not corrected with oversampling (e.g., SMOTE)
- No artifact rejection or ICA was applied
- Models used default hyperparameters (no tuning)
- Generalization across subjects was not explicitly evaluated