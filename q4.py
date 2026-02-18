import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

#Loads data and also makes X and y

kidney_data = pd.read_csv("kidney_disease.csv")


X = kidney_data.drop("classification", axis=1)


y = kidney_data["classification"]

#Splits into train and test with fixed random_state

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

#this Trains KNN
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

#this Predicts Test data

y_predict = knn_model.predict(X_test)

#this is the Confusion matrix

confusion_matrix = confusion_matrix(y_test, y_predict)


accuracy = accuracy_score(y_test, y_predict)

precision = precision_score(y_test, y_predict)

recall = recall_score(y_test, y_predict)

f1 = f1_score(y_test, y_predict)

# this prints results
print(f"Confusion Matrix: {confusion_matrix}" )
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
# In kidney disease prediction:
# A True positive happens when the model properly predicts that the patient has kidney disease.

# A True Negative happens when the model properly predicts that the patient  does not have kidney disease

# A False Positive happens when the model predicts kidney disease for a  healthy patient which could lead to uneeded  tests.

# A False Negative happens when the model predicts no disease for a patient who  has the disease, this can be delay  treatment and be dangerous.

# accuracy alone may not be enough to evaluate a classification model since it only calculates the percentage of good predictions without taking into account any errors being made.

# Recall or Sensitivity is a very important metric missing, since if a kidney disease case gets bad, recall will measure how many  kidney disease cases the model properly finds and identifies.
