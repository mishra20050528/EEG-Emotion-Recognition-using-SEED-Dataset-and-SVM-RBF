import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# DATASET PATH
# =====================================================

DATASET_PATH = r"E:\downloads\ExtractedFeatures_1s"

# =====================================================
# LABELS
# =====================================================

seed_labels = np.array([
     1, 0,-1,-1, 0,
     1,-1, 0, 1, 1,
     0,-1, 0, 1,-1
])

label_map = {-1:0, 0:1, 1:2}
seed_labels = np.array([label_map[x] for x in seed_labels])

# =====================================================
# LOAD DATA
# =====================================================

X = []
y = []

files = sorted([
    f for f in os.listdir(DATASET_PATH)
    if f.endswith(".mat") and f != "label.mat"
])

print("Loading files...")

for file in files:

    mat = sio.loadmat(os.path.join(DATASET_PATH,file))

    for trial in range(15):

        key = f"de_movingAve{trial+1}"

        if key not in mat:
            continue

        data = mat[key]

        # Shape: (5, samples, 62)
        data = np.transpose(data,(1,2,0))

        samples = data.shape[0]

        data = data.reshape(samples,-1)

        X.append(data)

        labels = np.full(samples,seed_labels[trial])

        y.append(labels)

X = np.concatenate(X,axis=0)
y = np.concatenate(y,axis=0)

print("\nDataset Loaded")
print("X shape:",X.shape)
print("y shape:",y.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# PCA
# =====================================================

pca = PCA(n_components=0.95)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("\nFeatures after PCA:",X_train.shape[1])

# =====================================================
# SVM MODEL
# =====================================================

svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced'
)

print("\nTraining SVM...")

svm.fit(X_train,y_train)

# =====================================================
# PREDICTION
# =====================================================

y_pred = svm.predict(X_test)

# =====================================================
# METRICS
# =====================================================

acc = accuracy_score(y_test,y_pred)

prec = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

rec = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print("\n==============================")
print("RESULTS")
print("==============================")

print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ]
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[
        'Negative',
        'Neutral',
        'Positive'
    ],
    yticklabels=[
        'Negative',
        'Neutral',
        'Positive'
    ]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(svm,"seed_svm.pkl")
joblib.dump(scaler,"seed_scaler.pkl")
joblib.dump(pca,"seed_pca.pkl")

print("\nModel Saved Successfully")