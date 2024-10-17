
from abc import abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model:
    """This is an abstract class that represents a model."""

    def __init__(self) -> None:
        """Initialize an empty dictionary for parameters."""
        self._parameters = {}

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the data.

        :param np.ndarray observations: A 2D numpy array where each row represents
            an observation and each column represents a feature.
        :param np.ndarray ground_truth: A 1D numpy array containing the ground truth
        return: None
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        :param np.ndarray observations: A 2D numpy array where each row represents
            an observation and each column represents a feature.
        return: np.ndarray predictions: A 1D numpy array containing
            the predicted values
        """
        pass
    
