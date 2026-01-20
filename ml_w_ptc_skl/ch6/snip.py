import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score, 
                                     learning_curve, validation_curve, GridSearchCV, RandomizedSearchCV)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, 
                             matthews_corrcoef, make_scorer, roc_curve, auc)

# ==========================================
# 1. DATA LOADING & PREPROCESSING (ESSENTIAL)
# ==========================================
def load_and_prep_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    df = pd.read_csv(url, header=None)
    
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    
    # Encode labels (M -> 1, B -> 0)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Stratified split ensures class balance is maintained in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=1
    )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_prep_data()


# ==========================================
# 2. PIPELINE & BASELINE EVALUATION
# ==========================================
# Always use pipelines to avoid data leakage during scaling/PCA
pipe_lr = make_pipeline(StandardScaler(), 
                        PCA(n_components=2), 
                        LogisticRegression(random_state=1))

# Simple Hold-out method
pipe_lr.fit(X_train, y_train)
print(f"Test Accuracy (Hold-out): {pipe_lr.score(X_test, y_test):.3f}")

# Cross-Validation (More robust estimation)
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
print(f"CV Accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")


# ==========================================
# 3. HYPERPARAMETER TUNING (CRITICAL)
# ==========================================
# Use this block to find the best settings for your model
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Define grid: Linear kernel vs RBF kernel
param_grid = [
    {'svc__C': param_range, 'svc__kernel': ['linear']},
    {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}
]

# Optimize for 'accuracy' (or 'f1', 'precision', etc.)
gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10, 
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

print(f"Best CV Score: {gs.best_score_:.3f}")
print(f"Best Parameters: {gs.best_params_}")

# Evaluate best model on test set
clf = gs.best_estimator_
print(f"Test Score of Best Model: {clf.score(X_test, y_test):.3f}")


# ==========================================
# 4. DETAILED METRICS & CONFUSION MATRIX
# ==========================================
y_pred = clf.predict(X_test)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("\nConfusion Matrix:\n", confmat)

print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(f"MCC:       {matthews_corrcoef(y_test, y_pred):.3f}")


# ==========================================
# 5. VISUALIZATION (ANALYSIS TOOLS)
# ==========================================
def plot_learning_curve(estimator, X, y):
    """Diagnose bias vs variance"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator, X=X, y=y, 
        train_sizes=np.linspace(0.1, 1.0, 10), 
        cv=10, n_jobs=-1)
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_mean, 'bo-', label='Training acc')
    plt.plot(train_sizes, test_mean, 'gs--', label='Validation acc')
    plt.title("Learning Curve")
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Uncomment to run visualizations:
# plot_learning_curve(pipe_lr, X_train, y_train)