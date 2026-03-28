import pytest
from engine.simulator import MonteCarloSimulator, SimulationConfig


def test_simulator_output_shape():
    sim = MonteCarloSimulator(
        SimulationConfig(start_value=100, drift=0.001, volatility=0.02, n_sims=500)
    )
    outcomes = sim.run()
    assert len(outcomes) == 500


def test_invalid_start_value_raises():
    with pytest.raises(ValueError):
        MonteCarloSimulator(
            SimulationConfig(start_value=-1, drift=0.001, volatility=0.02)
        )
