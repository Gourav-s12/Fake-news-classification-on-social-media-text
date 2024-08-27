# -*- coding: utf-8 -*-
"""evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1--j_Wm7GcJoZI-d7n8OJ3jozEtiADFAr
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
# !pip install fastText
import fasttext

"""# Task 5"""

filename = "knn_model.pickle"
models = ["log_model", "knn_model", "kmean_model", "nn_model", "svm_model"]

for model_file in models:
    # Load the model
    with open(model_file + ".pickle", 'rb') as f:
        loaded_model = pickle.load(f)

    # Load test data
    with open('tfidf_test_vectors_with_labels.pickle', 'rb') as f:
        tfidf_test_vectors, test_labels = pickle.load(f)

    test_labels = test_labels.values.flatten()
    # Make predictions
    test_predictions = loaded_model.predict(tfidf_test_vectors)
    y_true = test_labels

    # Print confusion matrix
    print(f"{model_file} Confusion Matrix:")
    # print(confusion_matrix(y_true, test_predictions))
    print("\n")

    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, test_predictions)
    print(f"{model_file} Accuracy: {accuracy:.4f}")

    # Calculate and print precision, recall, and f1-score
    precision = precision_score(y_true, test_predictions, average='weighted')
    recall = recall_score(y_true, test_predictions, average='weighted')
    f1 = f1_score(y_true, test_predictions, average='weighted')

    print(f"{model_file} Precision: {precision:.4f}")
    print(f"{model_file} Recall: {recall:.4f}")
    print(f"{model_file} F1-Score: {f1:.4f}")

    # Print the classification report
    print(f"{model_file} Classification Report:")
    print(classification_report(y_true, test_predictions))
    print("\n")

"""## for fasttext"""

model = fasttext.load_model("model_fasttext.bin")
test_df=pd.read_csv('./test_split.csv')

test_predictions = [model.predict(text)[0][0] for text in test_df['tweet']]
test_predictions = [1 if label == '__label__real' else 0 for label in test_predictions]

# Print confusion matrix
print(f"fasttext Confusion Matrix:")
# print(confusion_matrix(test_df['label'], test_predictions))
print("\n")

# Calculate and print accuracy
accuracy = accuracy_score(test_df['label'], test_predictions)
print(f"fasttext Accuracy: {accuracy:.4f}")

# Calculate and print precision, recall, and f1-score
precision = precision_score(test_df['label'], test_predictions, average='weighted')
recall = recall_score(test_df['label'], test_predictions, average='weighted')
f1 = f1_score(test_df['label'], test_predictions, average='weighted')

print(f"fasttext Precision: {precision:.4f}")
print(f"fasttext Recall: {recall:.4f}")
print(f"fasttext F1-Score: {f1:.4f}")

# Print the classification report
print(f"fasttext Classification Report:")
print(classification_report(test_df['label'], test_predictions))
print("\n")

