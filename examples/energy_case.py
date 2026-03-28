from engine import MonteCarloSimulator, SimulationConfig, DecisionEngine, DecisionConfig, risk_metrics


def main() -> None:
    current = 1.0
    simulator = MonteCarloSimulator(
        SimulationConfig(
            start_value=current,
            drift=0.0030,
            volatility=0.0350,
            steps=24,
            n_sims=10000,
            random_seed=7,
            shock_distribution="student_t",
            student_t_df=3,
        )
    )
    outcomes = simulator.run()

    engine = DecisionEngine(
        DecisionConfig(action_cost=0.05, confidence_threshold=0.60, risk_aversion=0.75)
    )
    decision = engine.evaluate(current, outcomes)
    risk = risk_metrics(current, outcomes)

    print("Decision:", decision)
    print("Risk:", risk)


if __name__ == "__main__":
    main()
