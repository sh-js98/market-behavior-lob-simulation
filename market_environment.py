import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

START_CASH = 10000.0
START_INVENTORY = 10
ORDER_QTY = 2


# --- LOB STRUCTURES ---
@dataclass
class Order:
    agent_id: int
    side: str  # 'buy' or 'sell'
    price: float
    quantity: int
    duration: int


class LimitOrderBook:
    def __init__(self):
        self.bids: List[Order] = []
        self.asks: List[Order] = []
        self.last_price: float = 100.0
        self.history_prices = []
        self.history_quoted_spread = []  # best ask - best bid (after matching)
        self.history_mean_spread = []    # |mean(asks) - mean(bids)| (before matching)
        self.history_volume = []

    def add_order(self, order: Order):
        if order.side == 'buy':
            self.bids.append(order)
            self.bids.sort(key=lambda x: x.price, reverse=True)
        else:
            self.asks.append(order)
            self.asks.sort(key=lambda x: x.price)

    def quoted_spread(self) -> Optional[float]:
        """Best ask minus best bid. NaN (not 0) when one side of the book is empty."""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return np.nan

    def mean_spread(self) -> Optional[float]:
        """Spread as defined in Lussange et al. (2024): absolute difference
        between the mean of all bids and the mean of all asks in the book."""
        if self.bids and self.asks:
            return abs(np.mean([o.price for o in self.asks]) - np.mean([o.price for o in self.bids]))
        return np.nan

    def match_orders(self):
        self.history_mean_spread.append(self.mean_spread())  # full book, before clearing
        trades = []
        volume_step = 0
        while self.bids and self.asks and self.bids[0].price >= self.asks[0].price:
            best_bid, best_ask = self.bids[0], self.asks[0]
            trade_qty = min(best_bid.quantity, best_ask.quantity)
            trade_price = best_ask.price
            self.last_price = trade_price
            volume_step += trade_qty
            trades.append((best_bid.agent_id, best_ask.agent_id, trade_price, trade_qty))
            best_bid.quantity -= trade_qty
            best_ask.quantity -= trade_qty
            if best_bid.quantity == 0:
                self.bids.pop(0)
            if best_ask.quantity == 0:
                self.asks.pop(0)

        self.history_prices.append(self.last_price)
        self.history_quoted_spread.append(self.quoted_spread())
        self.history_volume.append(volume_step)
        return trades

    def update_durations(self):
        for o in self.bids + self.asks:
            o.duration -= 1
        self.bids = [o for o in self.bids if o.duration > 0]
        self.asks = [o for o in self.asks if o.duration > 0]


# --- AGENTS ---
class BaseTrader:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.cash = START_CASH
        self.inventory = START_INVENTORY
        self.prev_wealth = None

    def wealth(self, price: float) -> float:
        return self.cash + self.inventory * price


class QLearningTrader(BaseTrader):
    def __init__(self, agent_id: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        super().__init__(agent_id)
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}

    def get_state(self, lob: LimitOrderBook):
        return (round(lob.last_price, 0), self.inventory)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)  # 0: Hold, 1: Buy, 2: Sell
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        return int(np.argmax(self.q_table[state]))

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(3)
        old_val = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state][action] = old_val + self.alpha * (reward + self.gamma * next_max - old_val)


class HighAlphaTrader(QLearningTrader):
    """Overreacts strongly to recent outcomes via a high learning rate."""
    def __init__(self, agent_id: int, alpha: float = 0.85):
        super().__init__(agent_id, alpha=alpha, epsilon=0.1)


class HerdingTrader(QLearningTrader):
    """Follows the dominant market action of the previous step with high probability."""
    def __init__(self, agent_id: int, herd_probability: float = 0.75):
        super().__init__(agent_id)
        self.herd_probability = herd_probability

    def choose_action_herd(self, state, dominant_action: int):
        if random.random() < self.herd_probability:
            return dominant_action
        return self.choose_action(state)


class NoiseTrader(BaseTrader):
    """Places purely random orders and does not learn."""
    def get_state(self, lob: LimitOrderBook):
        return None

    def choose_action(self, state=None):
        return random.choice([0, 1, 2])


# --- SIMULATION ---
def build_population(n_standard=3, n_high_alpha=2, n_herding=2, n_noise=2):
    agents, i = [], 0
    for cls, n in [(QLearningTrader, n_standard), (HighAlphaTrader, n_high_alpha),
                   (HerdingTrader, n_herding), (NoiseTrader, n_noise)]:
        for _ in range(n):
            agents.append(cls(agent_id=i))
            i += 1
    return agents


def run_simulation(agents, steps: int = 150, seed: Optional[int] = None) -> LimitOrderBook:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    lob = LimitOrderBook()
    by_id = {a.agent_id: a for a in agents}
    dominant_action = 0  # 0: Hold, 1: Buy, 2: Sell

    for _ in range(steps):
        decisions = {}  # agent_id -> (state, action), kept per agent for the learning update
        for agent in agents:
            state = agent.get_state(lob)
            if isinstance(agent, HerdingTrader):
                action = agent.choose_action_herd(state, dominant_action)
            else:
                action = agent.choose_action(state)

            # Agents can only buy with enough cash and sell shares they hold
            if action == 1 and agent.cash < (lob.last_price + 1.0) * ORDER_QTY:
                action = 0
            if action == 2 and agent.inventory < ORDER_QTY:
                action = 0
            decisions[agent.agent_id] = (state, action)
            agent.prev_wealth = agent.wealth(lob.last_price)

            if action == 1:
                price = lob.last_price + random.choice([-0.5, 0.0, 0.5, 1.0])
                lob.add_order(Order(agent.agent_id, "buy", price, ORDER_QTY, duration=4))
            elif action == 2:
                price = lob.last_price + random.choice([-1.0, -0.5, 0.0, 0.5])
                lob.add_order(Order(agent.agent_id, "sell", price, ORDER_QTY, duration=4))

        actions = [a for _, a in decisions.values()]
        dominant_action = max(set(actions), key=actions.count)

        # Match orders and settle each trade in the agents' portfolios
        for buyer_id, seller_id, price, qty in lob.match_orders():
            by_id[buyer_id].cash -= price * qty
            by_id[buyer_id].inventory += qty
            by_id[seller_id].cash += price * qty
            by_id[seller_id].inventory -= qty

        # Each learning agent updates with ITS OWN state/action; reward = change in wealth
        for agent in agents:
            if isinstance(agent, QLearningTrader):
                state, action = decisions[agent.agent_id]
                reward = agent.wealth(lob.last_price) - agent.prev_wealth
                agent.update_q_value(state, action, reward, agent.get_state(lob))

        lob.update_durations()
    return lob


def summarize(lob: LimitOrderBook) -> dict:
    prices = np.array(lob.history_prices)
    returns = np.diff(np.log(prices))
    quoted = np.array(lob.history_quoted_spread, dtype=float)
    return {
        "quoted_spread": np.nanmean(quoted),
        "mean_spread": np.nanmean(np.array(lob.history_mean_spread, dtype=float)),
        "empty_book_share": np.mean(np.isnan(quoted)),  # steps where one side had no orders
        "volume": np.mean(lob.history_volume),
        "volatility": np.std(returns),
    }


def noise_trader_experiment(n_agents=50, fractions=(0.0, 0.2, 0.4, 0.6, 0.8), runs=20, steps=500):
    """Vary the share of noise traders (rest = standard Q-learners), as in Section 6 of the paper."""
    results = {}
    for p in fractions:
        n_noise = int(round(p * n_agents))
        stats = []
        for run in range(runs):
            agents = build_population(n_standard=n_agents - n_noise, n_high_alpha=0,
                                      n_herding=0, n_noise=n_noise)
            stats.append(summarize(run_simulation(agents, steps=steps, seed=run)))
        results[p] = {k: (np.mean([s[k] for s in stats]), np.std([s[k] for s in stats])) for k in stats[0]}
    return results


if __name__ == "__main__":
    # 1) Mixed-population run (original plot)
    lob = run_simulation(build_population(), steps=150, seed=42)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(lob.history_prices, color='purple', linewidth=1.5, label='Market Price Dynamics')
    axs[0].set_ylabel('Price ($)')
    axs[1].plot(lob.history_quoted_spread, color='darkorange', linewidth=1.2, label='Bid-Ask Spread')
    axs[1].set_ylabel('Spread ($)')
    axs[2].bar(range(len(lob.history_volume)), lob.history_volume, color='teal', alpha=0.7, label='Trading Volume')
    axs[2].set_xlabel('Simulation Step')
    axs[2].set_ylabel('Volume')
    for ax in axs:
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig('behavioral_experiment_results.png')
    print("Saved behavioral_experiment_results.png")

    # 2) Noise-trader experiment
    res = noise_trader_experiment()
    print("\nNoise-trader experiment (50 agents, 500 steps, mean +/- std over 20 runs)")
    print(f"{'noise %':>8} {'quoted spread':>16} {'mean spread':>16} {'empty book %':>13} {'volume':>14} {'volatility':>18}")
    for p, r in res.items():
        print(f"{p*100:8.0f} {r['quoted_spread'][0]:8.3f} +/- {r['quoted_spread'][1]:.3f}"
              f" {r['mean_spread'][0]:8.3f} +/- {r['mean_spread'][1]:.3f}"
              f" {r['empty_book_share'][0]*100:13.1f}"
              f" {r['volume'][0]:7.2f} +/- {r['volume'][1]:.2f}"
              f" {r['volatility'][0]:9.5f} +/- {r['volatility'][1]:.5f}")

    fracs = [p * 100 for p in res]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for key, label in [("quoted_spread", "Quoted (best ask - best bid)"), ("mean_spread", "Mean bids vs mean asks (paper)")]:
        axs[0].errorbar(fracs, [res[p][key][0] for p in res], yerr=[res[p][key][1] for p in res], marker='o', label=label)
    axs[0].set_title('Spread'); axs[0].legend()
    axs[1].errorbar(fracs, [res[p]["volume"][0] for p in res], yerr=[res[p]["volume"][1] for p in res], marker='o', color='teal')
    axs[1].set_title('Volume per step')
    axs[2].errorbar(fracs, [res[p]["volatility"][0] for p in res], yerr=[res[p]["volatility"][1] for p in res], marker='o', color='purple')
    axs[2].set_title('Return volatility')
    for ax in axs:
        ax.set_xlabel('Noise traders (%)'); ax.grid(True)
    plt.tight_layout()
    plt.savefig('noise_trader_experiment.png')
    print("Saved noise_trader_experiment.png")
