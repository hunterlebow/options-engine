import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any
import logging
from .abstract_volatility_model import AbstractVolatilityModel
from .utils import calculate_implied_volatility
from ..bsm import vectorized_bsm_price
from ..providers.fred import get_risk_free_rate

logger = logging.getLogger(__name__)

class SABRModel(AbstractVolatilityModel):
    """
    SABR (Stochastic Alpha, Beta, Rho) volatility model.
    Implements the AbstractVolatilityModel interface for seamless integration.
    """
    def _sabr_iv(self, K, F, t, alpha, beta, rho, nu):
        x = np.log(F / K)
        z = (nu / alpha) * (F * K)**((1 - beta) / 2) * x
        if abs(x) < 1e-8:
            return alpha / (F**(1 - beta))
        q = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        term1 = alpha / ((F * K)**((1 - beta) / 2))
        term2 = (1 + (((1 - beta)**2 / 24) * (alpha**2 / ((F * K)**(1-beta))) +
                      (rho * beta * nu * alpha) / (4 * (F * K)**((1-beta)/2)) +
                      (2 - 3 * rho**2) / 24 * nu**2) * t)
        return term1 * term2 * (z / q)

    def calibrate(self, options_data: pd.DataFrame) -> Dict[str, float]:
        df = calculate_implied_volatility(options_data)
        if len(df) < 4:
            logger.warning("SABR calibration requires at least 4 options with valid IV.")
            return None
        forward = df['underlying_price'].mean()
        time_to_expiry = df['dte'].mean() / 365.0
        def objective(params):
            alpha, beta, rho, nu = params
            model_vols = np.array([
                self._sabr_iv(K, forward, time_to_expiry, alpha, beta, rho, nu)
                for K in df['strike']
            ])
            return np.mean((model_vols - df['implied_vol'])**2)
        bounds = [(0.01, 2.0), (0.01, 1.0), (-0.99, 0.99), (0.01, 2.0)]
        x0 = [0.3, 0.5, 0.0, 0.5]
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        if result.success:
            return {'alpha': result.x[0], 'beta': result.x[1], 'rho': result.x[2], 'nu': result.x[3]}
        else:
            logger.error(f"SABR calibration failed: {result.message}")
            return None

    def predict(self, options_data: pd.DataFrame, params: Dict) -> np.ndarray:
        alpha, beta, rho, nu = params['alpha'], params['beta'], params['rho'], params['nu']
        risk_free_rate = get_risk_free_rate()
        sabr_vols = np.array([
            self._sabr_iv(row['strike'], row['underlying_price'], row['dte'] / 365.0, alpha, beta, rho, nu)
            for _, row in options_data.iterrows()
        ])
        prices = vectorized_bsm_price(
            S=options_data['underlying_price'].values,
            K=options_data['strike'].values,
            T=(options_data['dte'] / 365.0).values,
            r=risk_free_rate,
            sigma=sabr_vols,
            option_type=options_data['option_type'].values,
            q=0.0
        )
        return prices 