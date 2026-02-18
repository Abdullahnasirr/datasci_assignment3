import pandas as pd
from sklearn.model_selection import train_test_split

#loads the dataset

kidney_data = pd.read_csv("kidney_disease.csv")

#the matrix X is all column except the label column

X = kidney_data.drop("classification", axis=1)

#vector y is the classification column

y = kidney_data["classification"]

# 30% training, 70% training m fixed random_state

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)


# we shoudlnt train n test a model on the same data since the model will have bias and be more accurate, new data is needed for testing


# The testing set gives us a fair check on the model and its performance in real life.