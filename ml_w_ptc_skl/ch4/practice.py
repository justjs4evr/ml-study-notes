# practice.py
# Summary of Mushroom Dataset Classification (UCI ML Repo)
# Covers: OneHotEncoding, LabelEncoding, Troubleshooting Version Errors, and RandomForest

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading Mushroom dataset...")
mushroom = fetch_ucirepo(id=73)

# Separate features and targets
# X contains 22 categorical features (text data like 'x', 'b', 's')
X = mushroom.data.features
y = mushroom.data.targets

print(f"Original X Shape: {X.shape}") # (8124, 22)
print(f"Original y Shape: {y.shape}") # (8124, 1)

# ==========================================
# 2. PREPROCESSING: TARGET (LabelEncoder)
# ==========================================
# Issue encountered: y was a DataFrame column, but models prefer 1D arrays.
# Solution: Use LabelEncoder to turn 'p'/'e' into 1/0.

le = LabelEncoder()

# .ravel() flattens the array to avoid DataConversionWarning
y_encoded = le.fit_transform(y.values.ravel()) 

# Convert back to DataFrame for clean handling later
y_df = pd.DataFrame(y_encoded, columns=['Target'])
###print(f"Target encoded. Unique values: {y_df['Target'].unique()}") # [1, 0]

# ==========================================
# 3. PREPROCESSING: FEATURES (OneHotEncoder)
# ==========================================
# ERROR 1: TypeError: unexpected keyword argument 'sparse_output'
# Cause: Older sklearn versions use 'sparse=False', newer ones use 'sparse_output=False'.
# Fix: Used sparse=False to be safe.

# ERROR 2: ValueError: Shape of passed values implies X, indices imply Y
# Cause: Trying to put 95 new columns back into a DataFrame with only the original 22 names.
# Fix: Use get_feature_names_out() to generate the correct 95 column headers.

###print("Encoding features...")
encoder = OneHotEncoder(
    sparse=False,        # Use sparse_output=False for sklearn > 1.2
    drop='first',        # Prevents multicollinearity (dummy variable trap)
    handle_unknown='ignore'
)

# Transform the data
X_encoded_array = encoder.fit_transform(X)

# Get the new column names (Expands from 22 -> 95 features)
new_feature_names = encoder.get_feature_names_out(X.columns)

# Create the clean, numerical DataFrame
X_final = pd.DataFrame(X_encoded_array, columns=new_feature_names)

###print(f"Encoded X Shape: {X_final.shape}") # Should be (8124, 95)

# ==========================================
# 4. MODEL TRAINING
# ==========================================
# ERROR 3: ValueError: could not convert string to float: 'x'
# Cause: Passing raw 'X' (text) into train_test_split/fit instead of 'X_final' (numbers).
# Fix: Ensure X_final is passed to the split function.

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final,       # <--- MUST use the encoded data, not raw X
    y_df, 
    test_size=0.3, 
    stratify=y_df, # Keeps class balance consistent
    random_state=42
)

# Initialize Random Forest
forest = RandomForestClassifier(
    n_estimators=200, 
    max_depth=15, 
    n_jobs=2, 
    random_state=42
)

print("Training Random Forest...")
# y_train.values.ravel() converts the y DataFrame column into a clean 1D array
forest.fit(X_train, y_train.values.ravel())

# ==========================================
# 5. EVALUATION
# ==========================================
train_score = forest.score(X_train, y_train)
test_score = forest.score(X_test, y_test)

print("-" * 30)
print(f"Training Accuracy: {train_score}") # 1.0
print(f"Test Accuracy:     {test_score}")  # 1.0
print("-" * 30)