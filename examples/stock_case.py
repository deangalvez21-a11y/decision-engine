from engine import MonteCarloSimulator, SimulationConfig, DecisionEngine, DecisionConfig, risk_metrics


def main() -> None:
    current = 100.0
    simulator = MonteCarloSimulator(
        SimulationConfig(
            start_value=current,
            drift=0.0010,
            volatility=0.0200,
            steps=30,
            n_sims=10000,
            random_seed=42,
            shock_distribution="normal",
        )
    )
    outcomes = simulator.run()

    engine = DecisionEngine(
        DecisionConfig(action_cost=0.02, confidence_threshold=0.55, risk_aversion=0.50)
    )
    decision = engine.evaluate(current, outcomes)
    risk = risk_metrics(current, outcomes)

    print("Decision:", decision)
    print("Risk:", risk)


if __name__ == "__main__":
    main()
