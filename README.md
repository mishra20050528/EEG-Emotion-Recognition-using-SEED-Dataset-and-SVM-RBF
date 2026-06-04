# EEG-Based Emotion Recognition and Recommendation System using SVM

## Overview

This project recognizes human emotions from EEG brain signals using the SEED dataset. The model classifies emotions into three categories:

* Positive 😊
* Neutral 😐
* Negative 😔

Based on the predicted emotion, the system also provides simple recommendations such as music, learning content, or wellness activities.

---

## Dataset

**Dataset:** SEED (SJTU EEG Emotion Dataset)

* 15 Subjects
* 62 EEG Channels
* 3 Emotion Classes
* 3 Recording Sessions
* Sampling Frequency: 200 Hz

Emotion Labels:

| Label | Emotion  |
| ----- | -------- |
| -1    | Negative |
| 0     | Neutral  |
| 1     | Positive |

For this project, the **de_movingAve** feature from the ExtractedFeatures_1s dataset was used.

---

## Why de_movingAve?

* Based on Differential Entropy (DE)
* Reduces EEG noise using Moving Average smoothing
* Produces stable features
* Gives better classification performance

---

## Methodology

1. Load EEG features from the SEED dataset.
2. Reshape the data into feature vectors.
3. Split data into training and testing sets.
4. Apply StandardScaler for normalization.
5. Apply PCA to retain 95% variance.
6. Train an SVM classifier with RBF kernel.
7. Evaluate model performance.

Pipeline:

SEED Dataset → de_movingAve → StandardScaler → PCA → SVM → Emotion Prediction

---

## Model Configuration

### PCA

* Variance Retained: 95%

### SVM

* Kernel: RBF
* C = 10
* Gamma = scale

---

## Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 96.19% |
| Precision | 96.20% |
| Recall    | 96.19% |
| F1 Score  | 96.19% |

The confusion matrix shows that most samples are correctly classified with very few errors.

---

## Recommendation System

### Positive

* Motivational Music
* Action Movies
* Productivity Activities

### Neutral

* Educational Content
* News
* General Entertainment

### Negative

* Relaxing Music
* Meditation
* Wellness Activities

---

## Future Scope

* LOSO Validation
* EEGNet and Deep Learning Models
* Real-Time Emotion Recognition
* Mobile Application Deployment

---

## Applications

* Mental Health Monitoring
* Healthcare
* Smart Learning Systems
* Human-Computer Interaction
* Brain-Computer Interfaces

---

## Author

**Sidhant Swaroop**
Engineering Student
