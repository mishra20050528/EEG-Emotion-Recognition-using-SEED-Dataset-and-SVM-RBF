# EEG Emotion Recognition using SEED Dataset and SVM-RBF

This project focuses on EEG-based emotion recognition using machine learning techniques on the SEED (SJTU Emotion EEG Dataset) dataset. The main objective of the project is to classify human emotions into Positive, Neutral, and Negative categories using EEG brain signals. EEG (Electroencephalogram) signals contain valuable information about brain activity and are widely used in Brain-Computer Interface (BCI) applications, healthcare systems, stress monitoring, and human-computer interaction.

In this project, Differential Entropy (DE) moving average features from the SEED dataset are used for emotion classification. The EEG data contains 62 channels and multiple frequency bands representing different brain activities. Statistical feature engineering techniques such as Mean and Standard Deviation are applied to transform the EEG signals into fixed-length feature vectors suitable for machine learning models.

The preprocessing pipeline includes feature extraction, normalization using StandardScaler, and optional dimensionality reduction using Principal Component Analysis (PCA). The classification model used in this project is Support Vector Machine (SVM) with RBF kernel, which is effective for handling nonlinear EEG patterns.

To ensure proper subject-independent evaluation, Leave-One-Subject-Out (LOSO) cross-validation is implemented. In this approach, the model is trained on all subjects except one and tested on the unseen subject. This method provides a realistic evaluation of the model’s generalization capability.

The project also includes hyperparameter tuning using GridSearchCV to optimize model performance. Evaluation metrics such as Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are used to analyze the classification results.

This project demonstrates the application of machine learning and EEG signal processing in emotion recognition systems and provides a strong baseline for future improvements using advanced deep learning architectures such as CNN, EEGNet, and Transformer-based models.
