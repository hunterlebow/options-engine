import numpy as np
from .polynomial_smile import PolynomialSmile

class QuadraticSmile(PolynomialSmile):
    """
    Quadratic smile model: σ(m) = a + b(m-1) + c(m-1)²
    """
    degree = 2

    def _polynomial_iv(self, moneyness: np.ndarray, params: list) -> np.ndarray:
        a, b, c = params
        m_centered = moneyness - 1.0
        return a + b * m_centered + c * m_centered**2 