# test_model.py
import pytest
from model import train_and_evaluate  # Import the function you want to test

def test_train_and_evaluate():
    accuracy = train_and_evaluate()
    assert accuracy > 0.8  # Assert that the model accuracy is above 80%
