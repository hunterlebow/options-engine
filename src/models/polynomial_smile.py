import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict
import logging
from .abstract_volatility_model import AbstractVolatilityModel
from .utils import calculate_implied_volatility
from ..bsm import vectorized_bsm_price
from ..providers.fred import get_risk_free_rate

logger = logging.getLogger(__name__)

class PolynomialSmile(AbstractVolatilityModel):
    """
    Base class for polynomial-based volatility smile models (Quadratic, Cubic).
    """
    degree: int = 0
    
    def _polynomial_iv(self, moneyness: np.ndarray, params: list) -> np.ndarray:
        raise NotImplementedError

    def calibrate(self, options_data: pd.DataFrame) -> Dict[str, float]:
        df = calculate_implied_volatility(options_data)
        if len(df) < self.degree + 1:
            logger.warning(f"Polynomial (deg={self.degree}) calibration requires at least {self.degree+1} options.")
            return None
        df['moneyness'] = df['underlying_price'] / df['strike']
        def objective(params: list):
            model_vols = self._polynomial_iv(df['moneyness'].values, params)
            return np.mean((model_vols - df['implied_vol'])**2)
        initial_guess = [df['implied_vol'].mean()] + [0.0] * self.degree
        result = minimize(objective, initial_guess, method='L-BFGS-B')
        if result.success:
            param_names = ['a', 'b', 'c', 'd'][:self.degree + 1]
            return dict(zip(param_names, result.x))
        else:
            logger.error(f"Polynomial (deg={self.degree}) calibration failed: {result.message}")
            return None

    def predict(self, options_data: pd.DataFrame, params: Dict) -> np.ndarray:
        param_list = [params[k] for k in sorted(params.keys())]
        risk_free_rate = get_risk_free_rate()
        moneyness = options_data['underlying_price'] / options_data['strike']
        poly_vols = self._polynomial_iv(moneyness.values, param_list)
        prices = vectorized_bsm_price(
            S=options_data['underlying_price'].values,
            K=options_data['strike'].values,
            T=(options_data['dte'] / 365.0).values,
            r=risk_free_rate,
            sigma=poly_vols,
            option_type=options_data['option_type'].values,
            q=0.0
        )
        return prices 