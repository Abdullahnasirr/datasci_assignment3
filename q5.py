import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Loads the data

kidney_data = pd.read_csv("kidney_disease.csv")

#Creates for X and y

X = kidney_data.drop("classification", axis=1)

y = kidney_data["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

k_values = [1, 3, 5, 7, 9]


results = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_predict = knn_model.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    results.append((k, acc))

# k vs accuracy table

results_table = pd.DataFrame(results, columns=["k", "test_accuracy"])

print(results_table)

# finds the best k
best_row = results_table.loc[results_table["test_accuracy"].idxmax()]

best_k = int(best_row["k"])

best_acc = float(best_row["test_accuracy"])


print(f"Best k: {best_k}")

print(f"Best accuracy: {best_acc}")


# the KNN model felexibility is dependant on the changing of k. When k is small, the model wont look at very many neighbors and it will become sensitive to
# smaller changes, because the model may focus on specific training points rather then general patterns because of the fewer k values.
#smoother predictions and data thats less sensitive to local patterns come when a k value is large. however the model may fail to capture important differences between classes.


