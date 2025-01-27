import mlflow
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn
import joblib

# Load the Iris dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define the objective function for Optuna to optimize
def objective(trial):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    # Create and train the model with the suggested hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Create and run the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Number of trials for hyperparameter tuning

# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
print(f"Best accuracy: {study.best_value}")

# Log experiment with MLflow
def train_and_evaluate():
    # Use the best hyperparameters found by Optuna
    best_params = study.best_params
    model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log experiment with MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    
    # Save the trained model to model.pkl
    joblib.dump(model, "model.pkl")
    
    return accuracy

if __name__ == "__main__":
    print(f"Model Accuracy: {train_and_evaluate()}")
