"""
Defines the abstract base class for all volatility models.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict

class AbstractVolatilityModel(ABC):
    """
    Abstract Base Class for all volatility models.
    
    This class defines a standard interface for calibrating models to market data
    and using them to predict option prices. This ensures that all models can be
    used interchangeably within the pricing and backtesting engines.
    """
    
    @abstractmethod
    def calibrate(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calibrate the model's parameters to fit the provided market option prices.

        Parameters
        ----------
        options_data : pd.DataFrame
            A DataFrame containing option chain data. Must include columns for
            'mid_price', 'underlying_price', 'strike', 'dte', and 'option_type'.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the calibrated model parameters.
            Returns None if calibration fails.
        """
        pass

    @abstractmethod
    def predict(self, options_data: pd.DataFrame, params: Dict) -> np.ndarray:
        """
        Predict option prices using the calibrated model parameters.

        Parameters
        ----------
        options_data : pd.DataFrame
            A DataFrame of options for which to predict prices. Must include
            columns for 'underlying_price', 'strike', 'dte', and 'option_type'.
        params : Dict
            A dictionary of model parameters obtained from the calibrate method.

        Returns
        -------
        np.ndarray
            An array of predicted option prices for the input options data.
        """
        pass 