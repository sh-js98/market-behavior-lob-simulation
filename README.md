# Heterogeneous Multi-Agent Limit Order Book (LOB) Simulation

An agent-based financial simulation modeling order book dynamics, price discovery, and volatility regimes driven by heterogeneous behavioral trader archetypes using reinforcement learning.

![Market Dynamics](behavioral_experiment_results.png)

## Overview

This repository simulates a continuous Limit Order Book (LOB) populated by distinct agent archetypes to evaluate how behavioral traits impact market quality, bid-ask spreads, and price discovery.

This project is inspired by Lussange et al. (2024), who study the same three trader traits (learning rate, herding, and noise trading) using their SYMBA multi-agent reinforcement learning simulator. This is an independent, simplified implementation, not a replication: it uses Q-learning agent archetypes and is not calibrated to real market data.

### Agent Archetypes

- **Standard Q-Learning Traders ($\alpha = 0.1, \varepsilon = 0.2$):** Evaluate cumulative rewards to anchor market clearing prices toward fundamental value.

- **High-Alpha Traders ($\alpha = 0.85$):** Overreact strongly to short-term price shifts, inducing temporary mispricings.

- **Herding Traders ($P_{\text{herd}} = 0.75$):** Copy dominant market trends, driving directional momentum and volatility clustering.

- **Noise Traders ($\varepsilon = 1.0$):** Submit uncoordinated random orders, acting as continuous baseline liquidity providers.

## Key Empirical Findings

- **High Fundamental Correlation ($r = 0.9872$):** Demonstrates long-term price efficiency despite localized behavioral distortions.

- **Volatility Spikes:** Herding runs temporarily drain order book depth, spiking rolling volatility up to **0.0315**.

- **Liquidity Buffering:** Uncoordinated noise trading reduces order starvation and narrows average bid-ask spreads.

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Simulation

```bash
python market_environment.py
```

Outputs simulation plots as `behavioral_experiment_results.png`.

## References

Lussange J, Vrizzi S, Palminteri S, Gutkin B (2024). Mesoscale effects of trader learning behaviors in financial markets: A multi-agent reinforcement learning study. *PLoS ONE* 19(4): e0301141. https://doi.org/10.1371/journal.pone.0301141

The authors' original simulator (SYMBA, C++): https://github.com/johannlussange/symba
