"""
Severity classifier module for glaucoma severity prediction based on CDR.
This module provides the SeverityClassifier class used by joblib for loading the rule-based model.
"""
import numpy as np
from typing import Union


class SeverityClassifier:
    """
    Rule-based classifier for glaucoma severity based on CDR percentage.
    This class matches the structure of the model saved by joblib in your training notebook.
    """
    
    def predict(self, cdr_percent: Union[float, np.ndarray, list]) -> str:
        """
        Predicts severity (Mild, Moderate, Severe) based on the CDR percentage.

        Args:
            cdr_percent (Union[float, np.ndarray, list]): The Cup-to-Disc Ratio percentage.
                                                       Can be a single float or a list/array
                                                       (the first element will be used).

        Returns:
            str: The predicted severity label ("Mild", "Moderate", or "Severe").
        """
        # Handle various input formats from scikit-learn models
        if isinstance(cdr_percent, np.ndarray):
            # Handle 2D array from sklearn: [[value]]
            if cdr_percent.ndim == 2 and cdr_percent.shape[0] == 1:
                cdr_percent = float(cdr_percent[0, 0])
            # Handle 1D array: [value]
            elif cdr_percent.ndim == 1 and len(cdr_percent) > 0:
                cdr_percent = float(cdr_percent[0])
            else:
                cdr_percent = float(cdr_percent.item())
        elif isinstance(cdr_percent, (list, tuple)):
            if len(cdr_percent) > 0:
                cdr_percent = float(cdr_percent[0])
            else:
                return "Error: No CDR data for severity prediction."
        else:
            cdr_percent = float(cdr_percent)

        if cdr_percent < 50:
            return "Mild"
        elif 50 <= cdr_percent < 80:
            return "Moderate"
        else:
            return "Severe"
