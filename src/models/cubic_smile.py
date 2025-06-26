import numpy as np
from .polynomial_smile import PolynomialSmile

class CubicSmile(PolynomialSmile):
    """
    Cubic smile model: σ(m) = a + b(m-1) + c(m-1)² + d(m-1)³
    """
    degree = 3

    def _polynomial_iv(self, moneyness: np.ndarray, params: list) -> np.ndarray:
        a, b, c, d = params
        m_centered = moneyness - 1.0
        return a + b * m_centered + c * m_centered**2 + d * m_centered**3 