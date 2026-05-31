# EEG-Based Emotion Recognition and Recommendation System using SVM

> An EEG-based Emotion Recognition System built using the SEED Dataset, Differential Entropy Features (de_movingAve), Principal Component Analysis (PCA), and Support Vector Machine (SVM) for emotion classification and personalized recommendation generation.

---

## 📌 Overview

Human emotions significantly influence behavior, decision-making, learning ability, and overall well-being. Traditional emotion recognition methods based on facial expressions, speech, or text are often affected by external conditions and can be intentionally manipulated.

Electroencephalography (EEG) directly captures brain activity, making it a reliable source for emotion recognition. This project utilizes EEG signals from the **SEED Dataset** and applies machine learning techniques to classify emotions into:

- 😊 Positive
- 😐 Neutral
- 😔 Negative

The predicted emotional state is then used to generate personalized recommendations.

---

## 🎯 Project Objectives

- Perform emotion recognition using EEG signals.
- Classify emotions into Positive, Neutral, and Negative categories.
- Develop a robust machine learning pipeline.
- Evaluate model performance using multiple metrics.
- Generate emotion-based recommendations.
- Analyze the effectiveness of Differential Entropy features.

---

# 🧠 Dataset

## SEED Dataset (SJTU Emotion EEG Dataset)

The SEED Dataset is a benchmark dataset widely used in EEG-based emotion recognition research.

### Dataset Information

| Parameter | Value |
|------------|---------|
| Subjects | 15 |
| EEG Channels | 62 |
| Emotion Classes | 3 |
| Sessions | 3 |
| Sampling Frequency | 200 Hz |

### Emotion Labels

| Label | Emotion |
|---------|---------|
| -1 | Negative |
| 0 | Neutral |
| 1 | Positive |

---

# 📂 Dataset Version Used

## ExtractedFeatures_1s

Instead of processing raw EEG signals directly, this project uses pre-extracted EEG features.

### Why ExtractedFeatures_1s?

- EEG divided into 1-second windows.
- Features extracted beforehand.
- Reduces computational complexity.
- Faster model training.
- Better memory efficiency.

---

# 🔍 Feature Selection

The dataset provides multiple feature representations:

- DE
- de_movingAve
- de_LDS
- DASM
- DASM_movingAve
- RASM
- DCAU

## Selected Feature

```text
de_movingAve
```

### Why de_movingAve?

- Based on Differential Entropy (DE).
- Moving Average smoothing reduces EEG noise.
- Provides stable emotional representation.
- Produces realistic classification performance.
- Achieved approximately **96.19% Accuracy**.

---

# ⚡ Understanding EEG Features

Each trial feature has the shape:

```text
(5, Samples, 62)
```

Where:

| Dimension | Meaning |
|------------|----------|
| 5 | Frequency Bands |
| Samples | EEG Windows |
| 62 | EEG Channels |

---

## EEG Frequency Bands

| Band | Frequency Range |
|--------|----------------|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–14 Hz |
| Beta | 14–31 Hz |
| Gamma | 31–50 Hz |

These bands capture different brain activities and emotional responses.

---

# 🔄 Data Preprocessing

## Step 1: Load Features

Features are loaded from MATLAB `.mat` files.

---

## Step 2: Data Reshaping

Original shape:

```text
(5, Samples, 62)
```

Transpose:

```text
(Samples, 62, 5)
```

Flatten:

```text
(Samples, 310)
```

Because:

```text
62 EEG Channels × 5 Frequency Bands = 310 Features
```

---

## Step 3: Label Mapping

Original Labels:

```text
-1 → Negative
 0 → Neutral
 1 → Positive
```

Mapped Labels:

```text
0 → Negative
1 → Neutral
2 → Positive
```

Reason:

- Better compatibility with Scikit-Learn classifiers.

---

## Step 4: Train-Test Split

Dataset split:

```text
80% Training
20% Testing
```

Using:

```python
train_test_split()
```

---

## Step 5: Feature Scaling

### StandardScaler

Formula:

\[
z = \frac{x-\mu}{\sigma}
\]

Benefits:

- Mean = 0
- Standard Deviation = 1
- Faster convergence
- Better SVM performance

---

# 📉 Dimensionality Reduction

## Principal Component Analysis (PCA)

Configuration:

```python
PCA(n_components=0.95)
```

### Purpose

- Reduce feature dimensionality.
- Remove redundant information.
- Retain 95% of data variance.
- Reduce training time.
- Improve generalization.

---

# 🤖 Machine Learning Model

## Support Vector Machine (SVM)

Configuration:

```python
SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced'
)
```

### Why SVM?

- Effective for high-dimensional data.
- Excellent performance on EEG signals.
- Handles nonlinear relationships.
- Strong generalization capability.

---

# 🔗 Machine Learning Pipeline

```text
SEED Dataset
      │
      ▼
ExtractedFeatures_1s
      │
      ▼
de_movingAve Features
      │
      ▼
Data Reshaping
      │
      ▼
Label Mapping
      │
      ▼
Train-Test Split
      │
      ▼
StandardScaler
      │
      ▼
PCA (95%)
      │
      ▼
SVM (RBF Kernel)
      │
      ▼
Emotion Classification
      │
      ▼
Recommendation System
```

---

# 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

# 🏆 Results

| Metric | Value |
|----------|---------|
| Accuracy | 96.19% |
| Precision | 96.20% |
| Recall | 96.19% |
| F1 Score | 96.19% |

---

# 📈 Classification Performance

## Accuracy

```text
96.19%
```

## Precision

```text
96.20%
```

## Recall

```text
96.19%
```

## F1 Score

```text
96.19%
```

---

# 🔥 Confusion Matrix

Add your confusion matrix image here:

```markdown
![Confusion Matrix](results/confusion_matrix.png)
```

### Observation

- Most values concentrated along the diagonal.
- Very few misclassifications.
- Strong emotion classification performance.

---

# 🎵 Emotion-Based Recommendation System

The predicted emotion is used to generate personalized recommendations.

## 😊 Positive Emotion

- Motivational Music
- Action Movies
- Productivity Tasks
- Skill Development Activities

---

## 😐 Neutral Emotion

- Educational Content
- News Articles
- Documentaries
- General Entertainment

---

## 😔 Negative Emotion

- Relaxing Music
- Meditation
- Wellness Activities
- Stress Relief Content

---

# 📁 Project Structure

```text
EEG-Emotion-Recognition/
│
├── dataset/
│   └── ExtractedFeatures_1s/
│
├── models/
│   └── svm_model.pkl
│
├── notebooks/
│   └── emotion_recognition.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── metrics.png
│   └── classification_report.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluation.py
│   └── recommendation_system.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/EEG-Emotion-Recognition.git
```

Navigate to project folder:

```bash
cd EEG-Emotion-Recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

```text
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
joblib
```

Install manually:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn joblib
```

---

# 🚀 Running the Project

Update dataset path:

```python
DATASET_PATH = "path_to_ExtractedFeatures_1s"
```

Run:

```bash
python train_model.py
```

Outputs:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- Saved Model

---

# 🔮 Future Work

- Leave-One-Subject-Out (LOSO) Validation
- EEGNet Architecture
- CNN-LSTM Models
- Real-Time Emotion Recognition
- Adaptive Recommendation Engine
- Mobile Deployment
- Brain-Computer Interface Integration

---

# 💡 Applications

- Mental Health Monitoring
- Healthcare Systems
- Personalized Learning Platforms
- Human-Computer Interaction
- Adaptive Entertainment Systems
- Brain-Computer Interfaces

---

# 👨‍💻 Author

**Sidh**

Final Year Engineering Project

---

# 🙏 Acknowledgements

- Shanghai Jiao Tong University (SEED Dataset)
- Scikit-Learn
- NumPy
- SciPy
- Open Source EEG Research Community

---

## ⭐ Support

If you found this project useful, please consider giving this repository a star.

```bash
⭐ Star this repository
🍴 Fork this repository
```
