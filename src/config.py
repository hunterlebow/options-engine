"""Configuration and environment variable management for the options engine."""

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the options engine."""
    
    # API Configuration
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    
    # Mispricing Detection
    MISPRICING_THRESHOLD: float = float(os.getenv("MISPRICING_THRESHOLD", "0.05"))
    
    # Risk-free Rate Configuration
    RISK_FREE_RATE_SOURCE: Literal["FRED", "constant"] = os.getenv(
        "RISK_FREE_RATE_SOURCE", "FRED"
    )
    CONSTANT_RF_RATE: float = float(os.getenv("CONSTANT_RF_RATE", "0.05"))
    
    # Default Parameters
    DEFAULT_UNDERLYING = "SPY"
    DEFAULT_DAYS_TO_EXPIRY = 30
    DEFAULT_MIN_DAYS_TO_EXPIRY = 7
    DEFAULT_MAX_DAYS_TO_EXPIRY = 45
    
    def validate(self) -> None:
        """Validate the configuration settings."""
        if not self.POLYGON_API_KEY:
            raise ValueError("POLYGON_API_KEY must be set in .env file")
        
        if self.MISPRICING_THRESHOLD <= 0:
            raise ValueError("MISPRICING_THRESHOLD must be positive")
        
        if self.RISK_FREE_RATE_SOURCE not in ["FRED", "constant"]:
            raise ValueError("RISK_FREE_RATE_SOURCE must be 'FRED' or 'constant'")
        
        if self.CONSTANT_RF_RATE <= 0:
            raise ValueError("CONSTANT_RF_RATE must be positive")

# Create global config instance
config = Config()
config.validate() 