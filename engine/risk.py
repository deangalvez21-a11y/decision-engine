from typing import Dict
import numpy as np


def risk_metrics(current_value: float, outcomes: np.ndarray) -> Dict[str, float]:
    if current_value <= 0:
        raise ValueError("current_value must be positive")
    if outcomes.size == 0:
        raise ValueError("outcomes cannot be empty")

    returns = (outcomes - current_value) / current_value
    return {
        "volatility": float(np.std(returns)),
        "worst_case_return": float(np.min(returns)),
        "best_case_return": float(np.max(returns)),
        "value_at_risk_95": float(np.percentile(returns, 5)),
        "value_at_risk_99": float(np.percentile(returns, 1)),
    }
