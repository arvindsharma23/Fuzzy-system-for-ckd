import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm

def perform_two_proportion_z_test(x1, n1, x2, n2, alternative='two-sided'):
    """
    Performs a two-proportion z-test to compare the performance of two models.
    This function replicates the statistical methodology from the provided research paper.

    Args:
        x1 (int): Number of correct predictions for Model 1.
        n1 (int): Total number of samples for Model 1.
        x2 (int): Number of correct predictions for Model 2.
        n2 (int): Total number of samples for Model 2.
        alternative (str): Defines the alternative hypothesis ('two-sided', 'greater', or 'less').

    Returns:
        tuple: A tuple containing the z-score and the p-value.
    """
    # Calculate sample proportions [cite: 192, 196]
    p1_hat = x1 / n1
    p2_hat = x2 / n2

    # Calculate pooled sample proportion [cite: 198]
    p_pooled = (x1 + x2) / (n1 + n2)
    
    # Calculate the standard error of the difference
    standard_error = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

    # Calculate the z-score test statistic [cite: 200]
    # Handle case where standard_error is zero to avoid division by zero
    if standard_error == 0:
        return 0.0, 1.0
        
    z_score = (p1_hat - p2_hat) / standard_error

    # Calculate the p-value based on the alternative hypothesis
    if alternative == 'two-sided':
        p_value = 2 * norm.sf(abs(z_score))
    elif alternative == 'greater': # Ha: p1 > p2
        p_value = norm.sf(z_score)
    elif alternative == 'less': # Ha: p1 < p2
        p_value = norm.cdf(z_score)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    return z_score, p_value

# 1. Load and Prepare the Dataset
try:
    df = pd.read_csv('cleaned_kidney_disease.csv')
except FileNotFoundError:
    print("Error: 'cleaned_kidney_disease.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Define features (X) and target (y)
X = df.drop('class', axis=1)
y = df['class']

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the features for better performance, especially for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train and Evaluate Proxy Models

# Model 1: Logistic Regression (Proxy for the Fuzzy System)
model_1 = LogisticRegression(max_iter=1000, random_state=42)
model_1.fit(X_train_scaled, y_train)
y_pred_1 = model_1.predict(X_test_scaled)
acc_1 = accuracy_score(y_test, y_pred_1)

# Model 2: MLP Classifier (Proxy for the ANFIS Model)
model_2 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
model_2.fit(X_train_scaled, y_train)
y_pred_2 = model_2.predict(X_test_scaled)
acc_2 = accuracy_score(y_test, y_pred_2)

# 3. Gather Data for Statistical Test
n_test = len(y_test)
correct_preds_1 = int(acc_1 * n_test)
correct_preds_2 = int(acc_2 * n_test)

print("--- Model Performance Analysis ---")
print(f"Model 1 (Logistic Regression) Accuracy: {acc_1:.4f}")
print(f"  - Test Set Size (n1): {n_test}")
print(f"  - Correct Predictions (x1): {correct_preds_1}\n")

print(f"Model 2 (MLP Classifier) Accuracy: {acc_2:.4f}")
print(f"  - Test Set Size (n2): {n_test}")
print(f"  - Correct Predictions (x2): {correct_preds_2}\n")


# 4. Perform the Two-Proportion Z-Test
# Following the paper's hypothesis: check if Model 2 is significantly better than Model 1.
# Null Hypothesis (H0): p1 = p2 (Accuracies are equal) [cite: 181]
# Alternative Hypothesis (Ha): p1 < p2 (Model 2 accuracy is greater than Model 1) [cite: 184]
# This corresponds to the 'less' alternative in our function.

z_stat, p_val = perform_two_proportion_z_test(correct_preds_1, n_test, correct_preds_2, n_test, alternative='less')

# 5. Interpret the Results
alpha = 0.05  # Significance level [cite: 186]

print("--- Statistical Validation (Two-Proportion Z-Test) ---")
print("Null Hypothesis (H0): The classification accuracy of both models is the same.")
print("Alternative Hypothesis (Ha): The accuracy of Model 2 (MLP) is greater than Model 1 (Logistic Regression).\n")
print(f"Calculated Z-score: {z_stat:.4f}")
print(f"Calculated P-value: {p_val:.4f}\n")

print("--- Conclusion ---")
if p_val < alpha:
    print(f"Since the p-value ({p_val:.4f}) is less than the significance level ({alpha}), we REJECT the null hypothesis.")
    print("The observed difference in accuracy is statistically significant.")
    print("We can conclude that Model 2 (MLP Classifier) is significantly more accurate than Model 1 (Logistic Regression) for this dataset.")
else:
    print(f"Since the p-value ({p_val:.4f}) is greater than the significance level ({alpha}), we FAIL to reject the null hypothesis.")
    print("The observed difference in accuracy is NOT statistically significant.")
    print("The minor variation in performance could be due to random chance in the sample data.")