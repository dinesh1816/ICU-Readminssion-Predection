# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
#
#
# # Load the dataset
# data = pd.read_csv('FinalDataset2.csv')
#
# # Displaying the first few rows of the dataset
# print(data.head())
#
# # Summarizing the dataset to understand its structure
# data.info()
#
# # Converting 'vdate' and 'discharged' to datetime format
# data['vdate'] = pd.to_datetime(data['vdate'], errors='coerce')
# data['discharged'] = pd.to_datetime(data['discharged'], errors='coerce')
#
# # Sorting data by 'eid' and 'vdate'
# data.sort_values(by=['eid', 'vdate'], inplace=True)
#
# # Creating 'readmissions' column where any subsequent visit is considered a readmission
# data['readmissions'] = data.duplicated(subset=['eid'], keep='first').astype(int)
#
# # Extracting Date Features
# data['vdate_day'] = data['vdate'].dt.day
# data['vdate_month'] = data['vdate'].dt.month
# data['vdate_year'] = data['vdate'].dt.year
#
# # Defining numerical and categorical features
# numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
# categorical_features = ['gender', 'facid']
#
# # Removing target and non-feature columns from numerical features
# numerical_features = [feature for feature in numerical_features if feature not in ['readmissions', 'eid', 'discharged']]
#
# # Adding date features to numerical features
# numerical_features += ['vdate_day', 'vdate_month', 'vdate_year']
#
# # Column Transformer for applying transformations to the appropriate columns
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_features),
#         ('cat', OneHotEncoder(), categorical_features)
#     ])
#
# # Defining features matrix X and target y
# X = data.drop(['readmissions', 'eid', 'vdate', 'discharged'], axis=1)
# y = data['readmissions']
#
# # Applying transformations
# X_processed = preprocessor.fit_transform(X)
#
# # Splitting data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
#
# # Initializing and training the Logistic Regression model
# lr_model = LogisticRegression()
# lr_model.fit(X_train, y_train)
#
# # Making predictions and evaluate the Logistic Regression model
# lr_y_pred = lr_model.predict(X_test)
# lr_y_prob = lr_model.predict_proba(X_test)[:, 1]  # probabilities for ROC AUC
#
# # Evaluation metrics for Logistic Regression
# lr_accuracy = accuracy_score(y_test, lr_y_pred)
# lr_precision = precision_score(y_test, lr_y_pred)
# lr_recall = recall_score(y_test, lr_y_pred)
# lr_f1 = f1_score(y_test, lr_y_pred)
# lr_roc_auc = roc_auc_score(y_test, lr_y_prob)
#
# # Printing Logistic Regression evaluation metrics
# print("\nLogistic Regression Metrics:")
# print(f"Accuracy: {lr_accuracy:.2f}")
# print(f"Precision: {lr_precision:.2f}")
# print(f"Recall: {lr_recall:.2f}")
# print(f"F1 Score: {lr_f1:.2f}")
# print(f"ROC AUC Score: {lr_roc_auc:.2f}")
#
# # Initializing and train the Random Forest model
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Making predictions and evaluate the Random Forest model
# rf_y_pred = rf_model.predict(X_test)
# rf_y_prob = rf_model.predict_proba(X_test)[:, 1]  # probabilities for ROC AUC
#
# # Evaluating metrics for Random Forest
# rf_accuracy = accuracy_score(y_test, rf_y_pred)
# rf_precision = precision_score(y_test, rf_y_pred)
# rf_recall = recall_score(y_test, rf_y_pred)
# rf_f1 = f1_score(y_test, rf_y_pred)
# rf_roc_auc = roc_auc_score(y_test, rf_y_prob)
#
# # Printing Random Forest evaluation metrics
# print("\nRandom Forest Metrics:")
# print(f"Accuracy: {rf_accuracy:.2f}")
# print(f"Precision: {rf_precision:.2f}")
# print(f"Recall: {rf_recall:.2f}")
# print(f"F1 Score: {rf_f1:.2f}")
# print(f"ROC AUC Score: {rf_roc_auc:.2f}")
#
# # Applying Hyperparameter tuning for Logistic Regression
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear']  # 'liblinear' is a good choice for small datasets and supports l1 and l2 regularization.
# }
#
# grid_search_lr = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='roc_auc', verbose=1)
# grid_search_lr.fit(X_train, y_train)
#
# # Best model from grid search
# best_model = grid_search_lr.best_estimator_
#
# # Re-evaluating the best model
# y_pred_best = best_model.predict(X_test)
# y_prob_best = best_model.predict_proba(X_test)[:, 1]
#
# # Evaluation metrics for the best model
# accuracy_best = accuracy_score(y_test, y_pred_best)
# precision_best = precision_score(y_test, y_pred_best)
# recall_best = recall_score(y_test, y_pred_best)
# f1_best = f1_score(y_test, y_pred_best)
# roc_auc_best = roc_auc_score(y_test, y_prob_best)
#
# print(f"\nOptimized Logistic Regression Accuracy: {accuracy_best:.2f}")
# print(f"Optimized Logistic Regression Precision: {precision_best:.2f}")
# print(f"Optimized Logistic Regression Recall: {recall_best:.2f}")
# print(f"Optimized Logistic Regression F1 Score: {f1_best:.2f}")
# print(f"Optimized Logistic Regression ROC AUC Score: {roc_auc_best:.2f}")
#
# print("\nWe use Logistic Regression model, as it is best suitable for this project.\n")
#
# # Model interpretation for logistic regression.
# # Getting the feature names after one-hot encoding
# feature_names = numerical_features + preprocessor.named_transformers_['cat'].get_feature_names(categorical_features).tolist()
#
# # Coefficients from the Logistic Regression model
# coef_dict = {}
# for coef, feat in zip(best_model.coef_[0], feature_names):
#     coef_dict[feat] = coef
#
# # Displaying the coefficients sorted by their absolute values
# sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
# print("\nFeature Coefficients from Logistic Regression:")
# for feature, coef in sorted_coef:
#     print(f"{feature}: {coef}")
#
# # Model Validation
# # Performing cross-validation
# cv_scores = cross_val_score(best_model, X_processed, y, cv=5, scoring='roc_auc')
#
# # Printing the cross-validation scores
# print("\nCross-validation ROC AUC scores:", cv_scores)
# print("Mean ROC AUC score:", cv_scores.mean())
#
# # ROC Curve for Logistic Regression and Random Forest
# # Assuming you have lr_y_prob and rf_y_prob as the probability outputs for Logistic Regression and Random Forest
# fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_y_prob)
# roc_auc_lr = auc(fpr_lr, tpr_lr)
#
# fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_y_prob)
# roc_auc_rf = auc(fpr_rf, tpr_rf)
#
# plt.figure()
# plt.plot(fpr_lr, tpr_lr, color='darkorange', label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
# plt.plot(fpr_rf, tpr_rf, color='green', label='Random Forest (area = %0.2f)' % roc_auc_rf)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Comparison')
# plt.legend(loc="lower right")
# plt.show()
#
# # Precision-Recall Curve
# precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_y_prob)
# precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_y_prob)
#
# plt.figure()
# plt.plot(recall_lr, precision_lr, color='blue', label='Logistic Regression')
# plt.plot(recall_rf, precision_rf, color='red', label='Random Forest')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="lower left")
# plt.show()
#
# # Confusion Matrix Visualization
# cm_lr = confusion_matrix(y_test, lr_y_pred)
# cm_rf = confusion_matrix(y_test, rf_y_pred)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
# plt.title('Logistic Regression Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
#
# plt.subplot(1, 2, 2)
# sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
# plt.title('Random Forest Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
#
# plt.show()
# #
# # Cross-Validation Score Distribution
# sns.kdeplot(cv_scores, shade=True, color="olive")
# plt.title('Density Plot of Cross-Validation ROC AUC Scores')
# plt.xlabel('ROC AUC Score')
# plt.ylabel('Density')
# plt.show()
#
#
# # Save your model
# joblib.dump(lr_model, 'lr_model.pkl')  # Logistic Regression model
# joblib.dump(rf_model, 'rf_model.pkl')  # Random Forest model
#
# # Save preprocessors
# joblib.dump(preprocessor, 'preprocessor.pkl')  # This is your ColumnTransformer

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
#
# # Load the dataset
# data = pd.read_csv('FinalDataset2.csv')
#
# # Displaying the first few rows of the dataset
# print(data.head())
#
# # Summarizing the dataset to understand its structure
# data.info()
#
# # Converting 'vdate' and 'discharged' to datetime format
# data['vdate'] = pd.to_datetime(data['vdate'], errors='coerce')
# data['discharged'] = pd.to_datetime(data['discharged'], errors='coerce')
#
# # Sorting data by 'eid' and 'vdate'
# data.sort_values(by=['eid', 'vdate'], inplace=True)
#
# # Creating 'readmissions' column where any subsequent visit is considered a readmission
# data['readmissions'] = data.duplicated(subset=['eid'], keep='first').astype(int)
#
# # Extracting Date Features
# data['vdate_day'] = data['vdate'].dt.day
# data['vdate_month'] = data['vdate'].dt.month
# data['vdate_year'] = data['vdate'].dt.year
#
# # Defining numerical and categorical features
# numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
# categorical_features = ['gender', 'facid']
#
# # Removing target and non-feature columns from numerical features
# numerical_features = [feature for feature in numerical_features if feature not in ['readmissions', 'eid', 'discharged']]
#
# # Adding date features to numerical features
# numerical_features += ['vdate_day', 'vdate_month', 'vdate_year']
#
# # Column Transformer for applying transformations to the appropriate columns
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_features),
#         ('cat', OneHotEncoder(), categorical_features)
#     ])
#
# # Defining features matrix X and target y
# X = data.drop(['readmissions', 'eid', 'vdate', 'discharged'], axis=1)
# y = data['readmissions']
#
# # Applying transformations
# X_processed = preprocessor.fit_transform(X)
#
# # Splitting data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
#
# # Initializing and training the Logistic Regression model
# lr_model = LogisticRegression()
# lr_model.fit(X_train, y_train)
#
# # Making predictions and evaluate the Logistic Regression model
# lr_y_pred = lr_model.predict(X_test)
# lr_y_prob = lr_model.predict_proba(X_test)[:, 1]  # probabilities for ROC AUC
#
# # Evaluation metrics for Logistic Regression
# lr_accuracy = accuracy_score(y_test, lr_y_pred)
# lr_precision = precision_score(y_test, lr_y_pred)
# lr_recall = recall_score(y_test, lr_y_pred)
# lr_f1 = f1_score(y_test, lr_y_pred)
# lr_roc_auc = roc_auc_score(y_test, lr_y_prob)
#
# # Printing Logistic Regression evaluation metrics
# print("\nLogistic Regression Metrics:")
# print(f"Accuracy: {lr_accuracy:.2f}")
# print(f"Precision: {lr_precision:.2f}")
# print(f"Recall: {lr_recall:.2f}")
# print(f"F1 Score: {lr_f1:.2f}")
# print(f"ROC AUC Score: {lr_roc_auc:.2f}")
#
# # Initializing and train the Random Forest model
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Making predictions and evaluate the Random Forest model
# rf_y_pred = rf_model.predict(X_test)
# rf_y_prob = rf_model.predict_proba(X_test)[:, 1]  # probabilities for ROC AUC
#
# # Evaluating metrics for Random Forest
# rf_accuracy = accuracy_score(y_test, rf_y_pred)
# rf_precision = precision_score(y_test, rf_y_pred)
# rf_recall = recall_score(y_test, rf_y_pred)
# rf_f1 = f1_score(y_test, rf_y_pred)
# rf_roc_auc = roc_auc_score(y_test, rf_y_prob)
#
# # Printing Random Forest evaluation metrics
# print("\nRandom Forest Metrics:")
# print(f"Accuracy: {rf_accuracy:.2f}")
# print(f"Precision: {rf_precision:.2f}")
# print(f"Recall: {rf_recall:.2f}")
# print(f"F1 Score: {rf_f1:.2f}")
# print(f"ROC AUC Score: {rf_roc_auc:.2f}")
#
# # Applying Hyperparameter tuning for Logistic Regression
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear']  # 'liblinear' is a good choice for small datasets and supports l1 and l2 regularization.
# }
#
# grid_search_lr = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='roc_auc', verbose=1)
# grid_search_lr.fit(X_train, y_train)
#
# # Best model from grid search
# best_model = grid_search_lr.best_estimator_
#
# # Re-evaluating the best model
# y_pred_best = best_model.predict(X_test)
# y_prob_best = best_model.predict_proba(X_test)[:, 1]
#
# # Evaluation metrics for the best model
# accuracy_best = accuracy_score(y_test, y_pred_best)
# precision_best = precision_score(y_test, y_pred_best)
# recall_best = recall_score(y_test, y_pred_best)
# f1_best = f1_score(y_test, y_pred_best)
# roc_auc_best = roc_auc_score(y_test, y_prob_best)
#
# print(f"\nOptimized Logistic Regression Accuracy: {accuracy_best:.2f}")
# print(f"Optimized Logistic Regression Precision: {precision_best:.2f}")
# print(f"Optimized Logistic Regression Recall: {recall_best:.2f}")
# print(f"Optimized Logistic Regression F1 Score: {f1_best:.2f}")
# print(f"Optimized Logistic Regression ROC AUC Score: {roc_auc_best:.2f}")
#
# print("\nWe use Logistic Regression model, as it is best suitable for this project.\n")
#
# # Model interpretation for logistic regression.
# # Getting the feature names after one-hot encoding
# cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
# feature_names = numerical_features + list(cat_feature_names)
#
# # Coefficients from the Logistic Regression model
# coef_dict = {}
# for coef, feat in zip(best_model.coef_[0], feature_names):
#     coef_dict[feat] = coef
#
# # Displaying the coefficients sorted by their absolute values
# sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
# print("\nFeature Coefficients from Logistic Regression:")
# for feature, coef in sorted_coef:
#     print(f"{feature}: {coef}")
#
# # Model Validation
# # Performing cross-validation
# cv_scores = cross_val_score(best_model, X_processed, y, cv=5, scoring='roc_auc')
#
# # Printing the cross-validation scores
# print("\nCross-validation ROC AUC scores:", cv_scores)
# print("Mean ROC AUC score:", cv_scores.mean())
#
# # ROC Curve for Logistic Regression and Random Forest
# # Assuming you have lr_y_prob and rf_y_prob as the probability outputs for Logistic Regression and Random Forest
# fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_y_prob)
# roc_auc_lr = auc(fpr_lr, tpr_lr)
#
# fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_y_prob)
# roc_auc_rf = auc(fpr_rf, tpr_rf)
#
# plt.figure()
# plt.plot(fpr_lr, tpr_lr, color='darkorange', label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
# plt.plot(fpr_rf, tpr_rf, color='green', label='Random Forest (area = %0.2f)' % roc_auc_rf)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Comparison')
# plt.legend(loc="lower right")
# plt.show()
#
# # Precision-Recall Curve
# precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_y_prob)
# precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_y_prob)
#
# plt.figure()
# plt.plot(recall_lr, precision_lr, color='blue', label='Logistic Regression')
# plt.plot(recall_rf, precision_rf, color='red', label='Random Forest')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="lower left")
# plt.show()
#
# # Confusion Matrix Visualization
# cm_lr = confusion_matrix(y_test, lr_y_pred)
# cm_rf = confusion_matrix(y_test, rf_y_pred)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
# plt.title('Logistic Regression Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
#
# plt.subplot(1, 2, 2)
# sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
# plt.title('Random Forest Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
#
# plt.show()
# #
# # Cross-Validation Score Distribution
# sns.kdeplot(cv_scores, fill=True, color="olive")
# plt.title('Density Plot of Cross-Validation ROC AUC Scores')
# plt.xlabel('ROC AUC Score')
# plt.ylabel('Density')
# plt.show()
#
#
# # Save your model
# joblib.dump(lr_model, 'lr_model.pkl')  # Logistic Regression model
# joblib.dump(rf_model, 'rf_model.pkl')  # Random Forest model
#
# # Save preprocessors
# joblib.dump(preprocessor, 'preprocessor.pkl')  # This is your ColumnTransformer






import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Load the dataset
data = pd.read_csv('FinalDataset2.csv')
data.sort_values(by=['eid', 'vdate'], inplace=True)
data['readmissions'] = data.duplicated(subset='eid', keep=False).astype(int)

# Define features and target
selected_features = [
    'hematocrit', 'neutrophils', 'sodium', 'glucose',
    'bloodureanitro', 'creatinine', 'bmi', 'pulse',
    'respiration', 'lengthofstay'
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
param_grid_lr = {
    'C': [0.1, 1, 10],  # Reduced complexity
    'penalty': ['l2']
}

param_grid_rf = {
    'n_estimators': [100],  # Reduced number for quicker execution
    'max_depth': [10],  # Limit depth to prevent overfitting
    'min_samples_split': [5]  # Increase to reduce complexity
}

# Randomized Grid Search
random_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=5, scoring='roc_auc', verbose=1)
random_search_rf = RandomizedSearchCV(rf_model, param_grid_rf, n_iter=10, cv=5, scoring='roc_auc', verbose=1, random_state=42, n_jobs=-1)

# Fit models
random_search_lr.fit(X_train, y_train)
random_search_rf.fit(X_train, y_train)

# Select best model based on ROC AUC
best_model = random_search_lr.best_estimator_ if random_search_lr.best_score_ > random_search_rf.best_score_ else random_search_rf.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# Output results
print(f"Best Model: {type(best_model).__name__}")
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")

# Save model
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
