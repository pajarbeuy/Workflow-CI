import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


df = pd.read_csv("teen_social_media_preprocessed.csv")
X = df.drop("depression_label", axis=1)
y = df["depression_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mlflow.log_metric("accuracy",  accuracy_score(y_test, pred))
    mlflow.log_metric("f1_score",  f1_score(y_test, pred, average="weighted"))
    mlflow.log_metric("precision", precision_score(y_test, pred, average="weighted"))
    mlflow.log_metric("recall",    recall_score(y_test, pred, average="weighted"))

    mlflow.sklearn.log_model(model, "random_forest_model")

    print("Accuracy:", accuracy_score(y_test, pred))