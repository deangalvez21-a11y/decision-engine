# Decision Engine Under Uncertainty

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Literal
import numpy as np


ShockDistribution = Literal["normal", "student_t"]


@dataclass
class SimulationConfig:
    start_value: float
    drift: float
    volatility: float
    steps: int = 30
    n_sims: int = 10_000
    random_seed: int | None = 42
    shock_distribution: ShockDistribution = "normal"
    student_t_df: int = 3


class MonteCarloSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def _draw_terminal_shocks(self):
        if self.config.shock_distribution == "normal":
            return self.rng.normal(size=self.config.n_sims)

        raw = self.rng.standard_t(df=self.config.student_t_df, size=self.config.n_sims)
        scale = np.sqrt((self.config.student_t_df - 2) / self.config.student_t_df)
        return raw * scale

    def run(self):
        cfg = self.config
        z = self._draw_terminal_shocks()

        terminal = (
            (cfg.drift - 0.5 * cfg.volatility**2) * cfg.steps
            + cfg.volatility * np.sqrt(cfg.steps) * z
        )

        return cfg.start_value * np.exp(terminal)


@dataclass
class DecisionConfig:
    action_cost: float = 0.02
    confidence_threshold: float = 0.55
    risk_aversion: float = 0.5


class DecisionEngine:
    def __init__(self, config: DecisionConfig):
        self.config = config

    def evaluate(self, current_value: float, outcomes: np.ndarray) -> Dict[str, Any]:
        returns = (outcomes - current_value) / current_value

        mean_return = float(np.mean(returns))
        volatility = float(np.std(returns))
        prob_positive = float(np.mean(returns > 0))

        expected_value = mean_return - self.config.action_cost
        risk_adjusted = expected_value - self.config.risk_aversion * volatility

        decision = "TAKE ACTION" if (
            prob_positive > self.config.confidence_threshold
            and risk_adjusted > 0
        ) else "PASS"

        return {
            "decision": decision,
            "mean_return": mean_return,
            "volatility": volatility,
            "prob_positive": prob_positive,
            "expected_value": expected_value,
            "risk_adjusted_score": risk_adjusted
        }


def main():
    current = 100

    sim = MonteCarloSimulator(
        SimulationConfig(
            start_value=current,
            drift=0.001,
            volatility=0.02
        )
    )

    outcomes = sim.run()

    engine = DecisionEngine(DecisionConfig())
    result = engine.evaluate(current, outcomes)

    print(result)


if __name__ == "__main__":
    main()
