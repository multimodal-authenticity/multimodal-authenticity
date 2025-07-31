
from sklearn.metrics import accuracy_score, f1_score
import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MetricsTop:
    def __init__(self, mode):
        """
        Initialize the MetricsTop class with available metric evaluation functions.

        :param mode: This can be used to define different evaluation strategies based on the dataset or task.
        """
        # Dictionary mapping dataset names to their respective evaluation functions
        self.metrics_dict = {
            'Authenticity': self.__eval_regression
        }

    def __eval_regression(self, y_pred, y_true):
        """
        Evaluate regression performance using several metrics including MAE, accuracy, F1 score, and correlation.

        :param y_pred: List of predicted values (PyTorch tensors).
        :param y_true: List of true values (PyTorch tensors).
        :return: A dictionary containing evaluation metrics.
        """
        # Convert PyTorch tensors to numpy arrays
        y_pred = np.array([y.detach().numpy() for y in y_pred])
        y_true = np.array([y.detach().numpy() for y in y_true])

        # Mean Absolute Error (MAE)
        mae = np.mean(np.absolute(y_pred - y_true)).astype(np.float64)

        # Identify non-zero entries for calculating accuracy and F1 score
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        binary_truth = (y_true[non_zeros] > 0)  # Convert to binary
        binary_preds = (y_pred[non_zeros] > 0)  # Convert to binary

        # Accuracy and F1 Score for binary classification on non-zero entries
        acc2 = accuracy_score(binary_preds, binary_truth)
        f1_value = f1_score(binary_truth, binary_preds, average='weighted')

        # Flatten predictions and true values for correlation calculation
        y_pred = [i[0] for i in y_pred]
        y_true = [i[0] for i in y_true]
        corr = np.corrcoef(y_pred, y_true)[0][1]

        # Compile results into a dictionary with rounded values
        eval_results_reg = {
            "Acc_2": round(acc2, 4),
            "F1_score": round(f1_value, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
        }
        return eval_results_reg

    def getMetrics(self, datasetName):
        """
        Retrieve the appropriate evaluation function based on the dataset name.

        :param datasetName: The name of the dataset for which to retrieve the evaluation function.
        :return: The evaluation function for the given dataset.
        """
        return self.metrics_dict[datasetName]
