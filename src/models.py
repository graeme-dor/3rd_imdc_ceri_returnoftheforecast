import numpy as np
import pandas as pd

class DengueForecastingModel:
    """
    Base class representing the interface for models used in the Dengue 2026 challenge.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]

    def fit(self, df_train):
        """
        Fits model parameters on the training dataset.
        """
        pass

    def predict(self, df_target):
        """
        Generates probabilistic forecasts (quantiles) for target dates.
        Returns a DataFrame containing columns for 'uf', 'date', 'casos',
        and prediction quantiles ('q_0.025' to 'q_0.975').
        """
        raise NotImplementedError
