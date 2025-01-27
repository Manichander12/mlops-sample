# Use the official Python image from the Docker Hub
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory content into the container's /app directory
COPY . /app

# Install the necessary dependencies
RUN pip install flask joblib optuna scikit-learn mlflow

# Set the command to run the Flask app
CMD ["python", "app.py"]
