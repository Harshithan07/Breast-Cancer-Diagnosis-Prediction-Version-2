# Breast Cancer Diagnosis Prediction – Version 2 (Advanced Predictive Modeling on WDBC Dataset)


## ABOUT THE PROJECT:
This project is an advanced extension of the original breast cancer classification task, where the goal is to predict whether a tumor is malignant or benign using diagnostic features from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Building on the foundation of Version 1, this version integrates Support Vector Machines (SVM), advanced cross-validation techniques, detailed learning curve analysis, and business-relevant evaluation such as lift charts — enhancing the model selection process for high-stakes medical prediction tasks.


## USE CASE EXPLANATION:
In clinical environments, early detection of breast cancer is critical to patient outcomes. This project provides a data-driven decision support system to assist healthcare professionals by predicting tumor malignancy using easily extracted features. Belonging to the domain of healthcare analytics and bioinformatics, this model helps reduce misdiagnosis by focusing on recall (sensitivity), ensuring that malignant tumors are not missed. It also supports broader medical AI development, where high precision and model transparency are essential.


## HOW IT IS BUILT AND FULL WORKING:

1. Dataset: UCI WDBC dataset with 569 patient records and 30 diagnostic features (mean, SE, worst) extracted from tumor images.

2. Preprocessing:

- Removed ID field and encoded diagnosis label (B = 0, M = 1).

- Applied StandardScaler for all features to ensure fair model comparisons, especially for distance-based models like SVM and k-NN.

3. Modeling Techniques:

- Decision Tree: Explored max depth, min samples split, and Gini vs entropy.

- Logistic Regression: Compared L1 vs L2 penalties, tuned regularization strength.

- k-Nearest Neighbors: Tested different values of k (3, 5, 7, 9), distance metrics.

- Support Vector Machine (SVM): Used both linear and RBF kernels, varied C and gamma.

4. Hyperparameter Tuning:

- Used GridSearchCV (5-fold cross-validation) to optimize hyperparameters for each model.

- Compared models using precision, recall, F1-score, and ROC-AUC.

5. Model Evaluation:

- Generated confusion matrix, ROC curves, and lift charts to assess performance.

- Analyzed underfitting and overfitting using learning curves.

- Discussed trade-offs in model choice (e.g., Decision Trees are interpretable but prone to overfitting, while SVMs generalize better with tuning).

6. Business Relevance:

- Selected recall as primary metric, minimizing false negatives in medical prediction.

- Used lift curve analysis to simulate marketing-style ROI for correct cancer detection.


## OUTPUT AND RESULTS OR BENCHMARKS:

| Model                      | Accuracy | Recall | AUC   | F1 Score |
| -------------------------- | -------- | ------ | ----- | -------- |
| Logistic Regression (best) | 97%      | 100%   | 0.998 | 0.97     |
| SVM (RBF)                  | 96.5%    | 98%    | 0.997 | 0.96     |
| k-NN (k=7)                 | 93.3%    | 92%    | 0.982 | 0.93     |
| Decision Tree              | 92.1%    | 91%    | 0.948 | 0.92     |

- Logistic Regression with L1 penalty and liblinear solver emerged as the best model, offering perfect recall.

- SVM showed strong AUC and F1 with good generalization.

- k-NN performed well when scaled, but suffered when unscaled.

- Decision Tree showed signs of overfitting with deeper trees.


## SKILLS, TOOLS:
Python, scikit-learn, GridSearchCV, StandardScaler, Logistic Regression, k-NN, Decision Tree Classifier, Support Vector Machines, Matplotlib, Seaborn, confusion matrix, ROC/AUC analysis, lift curves, learning curve diagnostics


## KEYWORDS:
Breast cancer classification, predictive modeling, healthcare AI, medical diagnostics, SVM, logistic regression, model tuning, ROC curve, recall optimization, lift analysis, scikit-learn, supervised learning, Python ML
