import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data_clean/telco_clean.csv")

target_col = [c for c in df.columns if "Churn" in c][0]
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

mlflow.log_metric("accuracy", acc)
mlflow.sklearn.save_model(model, "docker_model")
mlflow.sklearn.log_model(model, "model")

print("CI Accuracy:", acc)
print("Model logged via MLflow Project.")