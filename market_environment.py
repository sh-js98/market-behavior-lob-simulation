import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

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
        self.history_spreads = []
        self.history_volume = []

    def add_order(self, order: Order):
        if order.side == 'buy':
            self.bids.append(order)
            self.bids.sort(key=lambda x: x.price, reverse=True)
        else:
            self.asks.append(order)
            self.asks.sort(key=lambda x: x.price)

    def get_spread(self):
        if self.bids and self.asks:
            return round(self.asks[0].price - self.bids[0].price, 2)
        return 0.0

    def match_orders(self):
        trades = []
        volume_step = 0
        while self.bids and self.asks and self.bids[0].price >= self.asks[0].price:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            
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
        self.history_spreads.append(self.get_spread())
        self.history_volume.append(volume_step)
        return trades

    def update_durations(self):
        for o in self.bids + self.asks:
            o.duration -= 1
        self.bids = [o for o in self.bids if o.duration > 0]
        self.asks = [o for o in self.asks if o.duration > 0]


# --- BASE Q-LEARNING TRADER ---
class QLearningTrader:
    def __init__(self, agent_id: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        self.agent_id = agent_id
        self.alpha = alpha       # Learning rate
        self.gamma = gamma       # Discount factor
        self.epsilon = epsilon   # Exploration rate
        self.cash = 10000.0
        self.inventory = 10
        self.q_table = {}
        self.last_action = 0

    def get_state(self, lob: LimitOrderBook):
        return (round(lob.last_price, 0), self.inventory)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)  # 0: Hold, 1: Buy, 2: Sell
        if random.random() < self.epsilon:
            action = random.choice([0, 1, 2])
        else:
            action = int(np.argmax(self.q_table[state]))
        self.last_action = action
        return action

    def update_q_value(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(3)
        old_val = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state][action] = old_val + self.alpha * (reward + self.gamma * next_max - old_val)


# --- BEHAVIORAL EXPERIMENT CLASSES ---

# 1. High Alpha Trader (Amplification / Overreaction)
class HighAlphaTrader(QLearningTrader):
    """Overreacts strongly to recent price changes via high alpha."""
    def __init__(self, agent_id: int, alpha: float = 0.85):
        super().__init__(agent_id, alpha=alpha, epsilon=0.1)


# 2. Herding Trader (Social Proof / Imitation)
class HerdingTrader(QLearningTrader):
    """Follows the dominant market action with high probability."""
    def __init__(self, agent_id: int, herd_probability: float = 0.75):
        super().__init__(agent_id)
        self.herd_probability = herd_probability

    def choose_action_herd(self, state, dominant_action: int):
        if random.random() < self.herd_probability:
            self.last_action = dominant_action
            return dominant_action
        return self.choose_action(state)


# 3. Noise Trader (Uncoordinated Liquidity)
class NoiseTrader:
    """Places uncoordinated, purely random orders (epsilon = 1.0)."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.cash = 10000.0
        self.inventory = 10
        self.last_action = 0

    def get_state(self, lob: LimitOrderBook):
        return None

    def choose_action(self, state=None):
        self.last_action = random.choice([0, 1, 2])
        return self.last_action

    def update_q_value(self, state, action, reward, next_state):
        pass  # Noise traders do not learn or update Q-values


# --- EXPERIMENT SIMULATION RUNNER ---
def run_behavioral_experiment(steps: int = 150):
    lob = LimitOrderBook()
    
    # Create heterogeneous agent population
    standard_agents = [QLearningTrader(agent_id=i, alpha=0.1) for i in range(1, 4)]
    high_alpha_agents = [HighAlphaTrader(agent_id=i, alpha=0.85) for i in range(4, 6)]
    herding_agents = [HerdingTrader(agent_id=i, herd_probability=0.75) for i in range(6, 8)]
    noise_agents = [NoiseTrader(agent_id=i) for i in range(8, 10)]

    all_agents = standard_agents + high_alpha_agents + herding_agents + noise_agents
    dominant_action = 0  # 0: Hold, 1: Buy, 2: Sell

    for step in range(1, steps + 1):
        recent_actions = []

        for agent in all_agents:
            state = agent.get_state(lob) if hasattr(agent, 'get_state') else None

            # Route action decision based on agent behavioral type
            if isinstance(agent, HerdingTrader):
                action = agent.choose_action_herd(state, dominant_action)
            elif isinstance(agent, NoiseTrader):
                action = agent.choose_action()
            else:
                action = agent.choose_action(state)

            recent_actions.append(action)

            # Order submission logic
            if action == 1:    # Buy Limit Order
                bid_price = lob.last_price + random.choice([-0.5, 0.0, 0.5, 1.0])
                lob.add_order(Order(agent.agent_id, "buy", bid_price, 2, duration=4))
            elif action == 2:  # Sell Limit Order
                ask_price = lob.last_price + random.choice([-1.0, -0.5, 0.0, 0.5])
                lob.add_order(Order(agent.agent_id, "sell", ask_price, 2, duration=4))

        # Identify dominant market direction for herding agents in next step
        if recent_actions:
            dominant_action = max(set(recent_actions), key=recent_actions.count)

        # Match orders
        trades = lob.match_orders()

        # Update learning agents
        for agent in all_agents:
            if isinstance(agent, QLearningTrader):
                pnl = agent.cash + (agent.inventory * lob.last_price)
                reward = pnl - 10000.0
                next_state = agent.get_state(lob)
                agent.update_q_value(state, action, reward, next_state)

        lob.update_durations()

    return lob


# --- MAIN EXECUTION & VISUALIZATION ---
if __name__ == "__main__":
    print("Running Behavioral Experiment Simulation (150 steps)...")
    lob_results = run_behavioral_experiment(steps=150)

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(lob_results.history_prices, color='purple', linewidth=1.5, label='Market Price Dynamics')
    axs[0].set_ylabel('Price ($)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(lob_results.history_spreads, color='darkorange', linewidth=1.2, label='Bid-Ask Spread')
    axs[1].set_ylabel('Spread ($)')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].bar(range(len(lob_results.history_volume)), lob_results.history_volume, color='teal', alpha=0.7, label='Trading Volume')
    axs[2].set_xlabel('Simulation Step')
    axs[2].set_ylabel('Volume')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('behavioral_experiment_results.png')
    print("Execution complete! Graph saved as 'behavioral_experiment_results.png'.")