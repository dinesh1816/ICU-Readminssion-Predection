import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import joblib

# Function to plot ROC Curve
# Function to plot ROC Curve
def plot_roc_curve(y_test, model_probs, model_name):
    fpr, tpr, _ = roc_curve(y_test, model_probs)
    plt.plot(fpr, tpr, linestyle='--', label=model_name)

# Function to plot Precision-Recall Curve
def plot_precision_recall_curve(y_test, model_probs, model_name):
    precision, recall, _ = precision_recall_curve(y_test, model_probs)
    plt.plot(recall, precision, marker='.', label=model_name)


# Load and prepare the dataset
data = pd.read_csv('FinalDataset2.csv')
data.sort_values(by=['eid', 'vdate'], inplace=True)
data['readmissions'] = data.duplicated(subset='eid', keep=False).astype(int)

# Define features and target
selected_features = [
    'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro',
    'creatinine', 'bmi', 'pulse', 'respiration', 'lengthofstay'
]
X = data[selected_features]
y = data['readmissions']

# Standardize features
preprocessor = StandardScaler()
X_processed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameter space
param_grid_lr = {'C': [0.1, 1, 10], 'penalty': ['l2']}
param_grid_rf = {'n_estimators': [100, 150], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}

# Randomized Grid Search
random_search_lr = RandomizedSearchCV(lr_model, param_grid_lr, n_iter=4, cv=5, scoring='roc_auc',
                                      verbose=1, random_state=42, n_jobs=2)
random_search_rf = RandomizedSearchCV(rf_model, param_grid_rf, n_iter=4, cv=5, scoring='roc_auc',
                                      verbose=1, random_state=42, n_jobs=2)

# Fit models
random_search_lr.fit(X_train, y_train)
random_search_rf.fit(X_train, y_train)

# Select and evaluate the best model
best_model = random_search_lr.best_estimator_ if random_search_lr.best_score_ > random_search_rf.best_score_ else random_search_rf.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# Output results
print(f"Best Model: {type(best_model).__name__}")
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")

# Save model and preprocessor
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Plotting
plt.figure(figsize=(10, 5))
plot_roc_curve(y_test, random_search_lr.predict_proba(X_test)[:, 1], 'Logistic Regression')
plot_roc_curve(y_test, random_search_rf.predict_proba(X_test)[:, 1], 'Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve Comparison')
plt.show()

plt.figure(figsize=(10, 5))
plot_precision_recall_curve(y_test, random_search_lr.predict_proba(X_test)[:, 1], 'Logistic Regression')
plot_precision_recall_curve(y_test, random_search_rf.predict_proba(X_test)[:, 1], 'Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curve Comparison')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix of Best Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

if isinstance(best_model, RandomForestClassifier):
    plt.figure(figsize=(10, 6))
    importances = best_model.feature_importances_
    forest_importances = pd.Series(importances, index=selected_features)
    forest_importances.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Importances')
    plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(random_search_lr.predict_proba(X_test)[:, 1], bins=10, kde=True, color='blue', label='Logistic Regression')
sns.histplot(random_search_rf.predict_proba(X_test)[:, 1], bins=10, kde=True, color='green', label='Random Forest')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.title('Histogram of Predicted Probabilities')
plt.show()
