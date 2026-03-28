from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass
class DecisionConfig:
    action_cost: float = 0.02
    confidence_threshold: float = 0.55
    risk_aversion: float = 0.50


class DecisionEngine:
    """Convert uncertain outcomes into an action recommendation."""

    def __init__(self, config: DecisionConfig):
        self.config = config

    def evaluate(self, current_value: float, outcomes: np.ndarray) -> Dict[str, Any]:
        if current_value <= 0:
            raise ValueError("current_value must be positive")
        if outcomes.size == 0:
            raise ValueError("outcomes cannot be empty")

        returns = (outcomes - current_value) / current_value
        mean_return = float(np.mean(returns))
        median_return = float(np.median(returns))
        volatility = float(np.std(returns))
        prob_positive = float(np.mean(returns > 0))
        downside_probability = float(np.mean(returns < 0))

        expected_value = mean_return - self.config.action_cost
        risk_adjusted_score = expected_value - self.config.risk_aversion * volatility

        decision = (
            "TAKE ACTION"
            if prob_positive >= self.config.confidence_threshold and risk_adjusted_score > 0
            else "PASS"
        )

        return {
            "decision": decision,
            "mean_return": mean_return,
            "median_return": median_return,
            "volatility": volatility,
            "prob_positive": prob_positive,
            "downside_probability": downside_probability,
            "expected_value": expected_value,
            "risk_adjusted_score": risk_adjusted_score,
        }
