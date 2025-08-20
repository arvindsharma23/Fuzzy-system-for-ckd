import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# --- 1. Load and Clean the Original Dataset ---
try:
    # Load the original raw data file
    df = pd.read_csv('kidney_disease.csv')
    print("Original dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'kidney_disease.csv' not found.")
    print("Please make sure the original data file is in the same directory.")
    exit()

# --- Data Cleaning and Preprocessing ---
# Drop the 'id' column as it's just an index
df.drop('id', axis=1, inplace=True)

# Correct column names for clarity
col_names = {
    'bp': 'blood_pressure', 'sg': 'specific_gravity', 'al': 'albumin', 'su': 'sugar',
    'rbc': 'red_blood_cells', 'pc': 'pus_cell', 'pcc': 'pus_cell_clumps', 'ba': 'bacteria',
    'bgr': 'blood_glucose_random', 'bu': 'blood_urea', 'sc': 'serum_creatinine',
    'sod': 'sodium', 'pot': 'potassium', 'hemo': 'hemoglobin', 'pcv': 'packed_cell_volume',
    'wc': 'white_blood_cell_count', 'rc': 'red_blood_cell_count', 'htn': 'hypertension',
    'dm': 'diabetes_mellitus', 'cad': 'coronary_artery_disease', 'appet': 'appetite',
    'pe': 'pedal_edema', 'ane': 'anemia', 'classification': 'class'
}
df.rename(columns=col_names, inplace=True)

# Clean and binarize the target column 'class'
df['class'] = df['class'].replace(to_replace={'ckd': 1, 'ckd\t': 1, 'notckd': 0})

# Convert columns that should be numeric, coercing errors to NaN
numeric_cols = [
    'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'hemoglobin'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values: Median for numerical, Mode for categorical
categorical_features = [col for col in df.columns if df[col].dtype == 'object']
numerical_features = [col for col in df.columns if df[col].dtype != 'object' and col != 'class']

# FIX: Flatten the output of the imputer to 1D to match the column's dimension
for col in numerical_features:
    df[col] = SimpleImputer(strategy='median').fit_transform(df[[col]]).ravel()
for col in categorical_features:
    df[col] = SimpleImputer(strategy='most_frequent').fit_transform(df[[col]]).ravel()

# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
print("Data cleaning and preprocessing complete.")


# --- 2. Prepare Data for Modeling ---
X = df.drop('class', axis=1)
y = df['class']

# --- 3. Split Data BEFORE Augmentation (Crucial Fix) ---
# This prevents data leakage. The test set will remain completely unseen.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Original data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

# --- 4. Augment ONLY the Training Data ---
# We will add synthetic data to the training set to reach 500 samples for training.
rows_to_add = 500 - len(X_train)
if rows_to_add > 0:
    X_train_aug = pd.concat([X_train, X_train.sample(n=rows_to_add, replace=True, random_state=42)], ignore_index=True)
    y_train_aug = pd.concat([y_train, y_train.sample(n=rows_to_add, replace=True, random_state=42)], ignore_index=True)
    print(f"Training data augmented to {len(X_train_aug)} rows.")
else:
    X_train_aug, y_train_aug = X_train, y_train


# --- 5. Feature Scaling ---
# Fit the scaler ONLY on the training data and transform both sets.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")

# --- 6. Train the Support Vector Machine (SVM) Model ---
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train_aug)
print("Support Vector Machine model training complete.")

# --- 7. Make Predictions on the UNTOUCHED Test Set ---
y_pred_actual = svm_model.predict(X_test_scaled)
y_pred_proba_actual = svm_model.predict_proba(X_test_scaled)[:, 1]

# --- 8. Display Actual Performance Metrics ---
print("\n--- Model Evaluation Results ---")
accuracy = accuracy_score(y_test, y_pred_actual) * 100
precision = precision_score(y_test, y_pred_actual) * 100
recall = recall_score(y_test, y_pred_actual) * 100
f1 = f1_score(y_test, y_pred_actual) * 100
auroc = roc_auc_score(y_test, y_pred_proba_actual)

print(f"Accuracy:  {accuracy:.1f}%")
print(f"Precision: {precision:.1f}%")
print(f"Recall:    {recall:.1f}%")
print(f"F1 Score:  {f1:.1f}%")
print(f"AUROC:     {auroc:.3f}")

print("\n--- End of Report ---")
