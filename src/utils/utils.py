"""
This python script contains utility or helper functions which might help in the project
"""

import zipfile

def unzip_data(filename: str,data_dir="data") -> None:
    """Uzips the files and folders into a directory

    Parameters
    ----------
    filename : str
        Name of the zip file
    data_dir : str, optional
        Directory you want the zip folder you want the files extracted into, by default "data"
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall(data_dir)
    zip_ref.close()

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from typing import Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
def get_metrics(y_true: np.array, y_pred: np.array, plot_conf_matrix: Optional[bool]=False) -> Dict[str, float]:
    """Returns classification metrics (includes accuracy, precision, recall) and optionally plots a confusion matrix

    Parameters
    ----------
    y_true : np.array
        True labels
    y_pred : np.array
        Predicted labels
    plot_conf_matrix : Optional[bool], optional
        Plots confusion matrix, by default False

    Returns
    -------
    Dict
        Contains metrics where key is the metric type
    """
    y_true, y_pred = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)
    metrics_dict = {'accuracy': float(accuracy_score(y_true, y_pred)), 'precision': float(precision_score(y_true, y_pred)), 'recall': float(recall_score(y_true, y_pred))}
    if plot_conf_matrix:
        cm = confusion_matrix(y_true, y_pred, normalize=True)
        sns.heatmap(cm, annot=True, cmap='Blues')
    return metrics_dict