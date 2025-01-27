import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn

def train_and_evaluate():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log experiment
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    return accuracy

if __name__ == "__main__":
    print(f"Model Accuracy: {train_and_evaluate()}")
