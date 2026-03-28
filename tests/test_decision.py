import numpy as np
from engine.decision import DecisionEngine, DecisionConfig


def test_decision_pass_for_bad_outcomes():
    engine = DecisionEngine(DecisionConfig())
    outcomes = np.array([70.0, 80.0, 90.0])
    result = engine.evaluate(100.0, outcomes)
    assert result["decision"] == "PASS"


def test_decision_take_action_for_good_outcomes():
    engine = DecisionEngine(DecisionConfig(action_cost=0.01, confidence_threshold=0.5, risk_aversion=0.1))
    outcomes = np.array([120.0, 130.0, 125.0, 118.0])
    result = engine.evaluate(100.0, outcomes)
    assert result["decision"] == "TAKE ACTION"
