# Decision Engine Under Uncertainty

A Python-based system that converts uncertain future outcomes into decisions using:

- Monte Carlo simulation
- Expected value logic
- Risk-adjusted scoring
- Configurable shock distributions (Normal / Student-t)

---

## Why this project exists

Most real-world systems are uncertain:

- Financial markets  
- Energy systems  
- Resource allocation  
- Strategic planning  

This engine provides a structured way to:
1. Simulate possible futures  
2. Quantify upside and downside  
3. Account for costs and risk  
4. Recommend whether to take action  

---

## Features

- Vectorized Monte Carlo simulation (NumPy)
- Memory-efficient terminal-value modeling
- Optional fat-tail modeling (Student-t distribution)
- Modular decision engine
- Risk metrics (volatility, VaR, worst/best case)

---

## Run the project

```bash
pip install -r requirements.txt
python decision_engine.py
```

---

## Example Use Cases

- Quantitative trading decisions  
- Energy investment modeling  
- Resource allocation under uncertainty  
- Strategic planning  

---

## What this demonstrates

- Probabilistic thinking  
- Simulation under uncertainty  
- Risk-aware decision-making  
- Clean system design  
