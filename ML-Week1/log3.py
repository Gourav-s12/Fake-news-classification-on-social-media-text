
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

with open('tfidf_train_vectors_with_labels.pickle', 'rb') as f:
    tfidf_train_vectors, train_labels = pickle.load(f)

with open('tfidf_val_vectors_with_labels.pickle', 'rb') as f:
    tfidf_val_vectors, val_labels = pickle.load(f)


# Assuming train_labels is a DataFrame column or another non-array type
train_labels = train_labels.values.flatten()
val_labels  = val_labels .values.flatten()

# Initialize and train Logistic Regression model
logreg_model = LogisticRegression(random_state=32)
logreg_model.fit(tfidf_train_vectors, train_labels)

# # Predictions on validation and test datasets
val_predictions_logreg = logreg_model.predict(tfidf_val_vectors)
# test_predictions_logreg = logreg_model.predict(tfidf_test_vectors)

# # Evaluation
print(val_predictions_logreg.shape)
print(val_labels.shape)
count = pd.Series(val_predictions_logreg).value_counts()
count2 = pd.Series(val_labels).value_counts()
print(count)
print(count2)
val_accuracy_logreg = accuracy_score(val_labels, val_predictions_logreg)
print("Validation Accuracy (Logistic Regression):", val_accuracy_logreg)

# test_accuracy_logreg = accuracy_score(test_df['label'], test_predictions_logreg)
# print("Test Accuracy (Logistic Regression):", test_accuracy_logreg)


# # Assuming tfidf_train_vectors and train_df['label'] are your TF-IDF vectors and labels for training
# parameters_logreg = {
#     'C': [0.001, 0.01, 0.1, 1],  # Regularization parameter
#     'penalty': ['l1', 'l2'],  # Regularization penalty
#     'solver': ['liblinear', 'saga'],  # Solver for optimization problem
# }

# logreg_model = LogisticRegression(random_state=42)

# grid_search_logreg = GridSearchCV(logreg_model, parameters_logreg, cv=5, scoring='accuracy')
# grid_search_logreg.fit(tfidf_train_vectors, train_labels)  # Fit GridSearchCV to the training data

# # Get the best hyperparameters
# best_hyperparameters_logreg = grid_search_logreg.best_params_
# best_model = grid_search_logreg.best_estimator_  # Get the best model
# print("Best Hyperparameters (Logistic Regression):", best_hyperparameters_logreg)

# val_predictions_log = best_model.predict(tfidf_val_vectors)
# # Evaluation
# val_accuracy_log = accuracy_score(val_labels, val_predictions_log)
# print("Validation Accuracy (Logistic Regression):", val_accuracy_log)

# # save model
# filename = "log_model.pickle"
# pickle.dump(best_model, open(filename, "wb"))