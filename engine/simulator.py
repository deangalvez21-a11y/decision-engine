from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
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
    """Monte Carlo simulator with memory-efficient terminal simulation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self._validate()

    def _validate(self) -> None:
        cfg = self.config
        if cfg.start_value <= 0:
            raise ValueError("start_value must be positive")
        if cfg.steps <= 0:
            raise ValueError("steps must be positive")
        if cfg.n_sims <= 0:
            raise ValueError("n_sims must be positive")
        if cfg.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if cfg.shock_distribution not in {"normal", "student_t"}:
            raise ValueError("shock_distribution must be 'normal' or 'student_t'")
        if cfg.shock_distribution == "student_t" and cfg.student_t_df <= 2:
            raise ValueError("student_t_df must be greater than 2")

    def _draw_terminal_shocks(self) -> np.ndarray:
        if self.config.shock_distribution == "normal":
            return self.rng.normal(size=self.config.n_sims)
        raw = self.rng.standard_t(df=self.config.student_t_df, size=self.config.n_sims)
        scale = np.sqrt((self.config.student_t_df - 2) / self.config.student_t_df)
        return raw * scale

    def _draw_path_shocks(self) -> np.ndarray:
        if self.config.shock_distribution == "normal":
            return self.rng.normal(size=(self.config.n_sims, self.config.steps))
        raw = self.rng.standard_t(
            df=self.config.student_t_df, size=(self.config.n_sims, self.config.steps)
        )
        scale = np.sqrt((self.config.student_t_df - 2) / self.config.student_t_df)
        return raw * scale

    def run(self) -> np.ndarray:
        """Return terminal values only."""
        cfg = self.config
        z = self._draw_terminal_shocks()
        terminal_log_return = (
            (cfg.drift - 0.5 * cfg.volatility**2) * cfg.steps
            + cfg.volatility * np.sqrt(cfg.steps) * z
        )
        return cfg.start_value * np.exp(terminal_log_return)

    def run_paths(self) -> np.ndarray:
        """Return full price paths for plotting and diagnostics."""
        cfg = self.config
        shocks = self._draw_path_shocks()
        log_returns = (
            (cfg.drift - 0.5 * cfg.volatility**2)
            + cfg.volatility * shocks
        )
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        return cfg.start_value * np.exp(cumulative_log_returns)
