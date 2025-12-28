"""
Online Learning with PPO (Proximal Policy Optimization)

This module implements a lightweight PPO agent for adaptive position sizing.
The agent learns optimal position sizes based on market conditions and trade outcomes.

Key Features:
- Continuous action space (position size 0-1)
- State: regime, volatility, signal strength, recent performance
- Safe policy updates with KL divergence constraints
- Experience replay with prioritized sampling
- Burn-in period before live deployment

Research basis:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Moody & Saffell, "Reinforcement Learning for Trading" (2001)
- Deng et al., "Deep Direct Reinforcement Learning for Financial Trading" (2016)

"The best position size is the one that maximizes long-term wealth,
not the one that maximizes any single trade."
"""

import logging
import json
import math
import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from collections import deque
import sqlite3

from .regime import MarketRegime, RegimeState

logger = logging.getLogger(__name__)


# Neural network simulation using simple function approximation
# In production, could be replaced with PyTorch/TensorFlow
class ActivationFunction(Enum):
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"


@dataclass
class NetworkWeights:
    """Simple neural network weights for policy and value functions"""
    # Input features -> Hidden layer (8 inputs, 16 hidden)
    w1: List[List[float]] = field(default_factory=list)
    b1: List[float] = field(default_factory=list)

    # Hidden -> Output layer (16 hidden, 1 output)
    w2: List[List[float]] = field(default_factory=list)
    b2: List[float] = field(default_factory=list)

    # Log standard deviation for action sampling
    log_std: float = -0.5

    def initialize(self, input_size: int = 8, hidden_size: int = 16):
        """Xavier initialization"""
        scale1 = math.sqrt(2.0 / (input_size + hidden_size))
        scale2 = math.sqrt(2.0 / (hidden_size + 1))

        self.w1 = [[random.gauss(0, scale1) for _ in range(hidden_size)]
                   for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size

        self.w2 = [[random.gauss(0, scale2)] for _ in range(hidden_size)]
        self.b2 = [0.0]

        self.log_std = -0.5

    def forward(self, x: List[float]) -> Tuple[float, float]:
        """Forward pass returning (mean, std)"""
        # Hidden layer
        hidden = []
        for j in range(len(self.b1)):
            h = sum(x[i] * self.w1[i][j] for i in range(len(x))) + self.b1[j]
            hidden.append(max(0, h))  # ReLU

        # Output layer
        mean = sum(hidden[j] * self.w2[j][0] for j in range(len(hidden))) + self.b2[0]
        mean = 1.0 / (1.0 + math.exp(-mean))  # Sigmoid for 0-1 range

        std = math.exp(self.log_std)

        return mean, std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "log_std": self.log_std,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkWeights":
        weights = cls()
        weights.w1 = data.get("w1", [])
        weights.b1 = data.get("b1", [])
        weights.w2 = data.get("w2", [])
        weights.b2 = data.get("b2", [])
        weights.log_std = data.get("log_std", -0.5)
        return weights


@dataclass
class Experience:
    """Single experience tuple for training"""
    state: List[float]          # Market state features
    action: float               # Position size taken
    reward: float               # Trade outcome
    next_state: List[float]     # State after trade
    done: bool                  # Episode ended
    log_prob: float             # Log probability of action
    value: float                # Value estimate
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context for debugging
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    regime: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "log_prob": self.log_prob,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "regime": self.regime,
        }


@dataclass
class PPOConfig:
    """PPO hyperparameters"""
    # Learning
    learning_rate: float = 0.0003
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda

    # PPO-specific
    clip_ratio: float = 0.2          # PPO clipping
    target_kl: float = 0.01          # KL divergence target
    entropy_coef: float = 0.01       # Entropy bonus
    value_coef: float = 0.5          # Value loss coefficient

    # Training
    batch_size: int = 32
    n_epochs: int = 4                # Epochs per update
    update_frequency: int = 50       # Trades between updates

    # Safety
    min_experiences: int = 100       # Burn-in period
    max_position_size: float = 1.0   # Maximum position size
    min_position_size: float = 0.1   # Minimum position size

    # Experience replay
    buffer_size: int = 10000
    prioritized: bool = True


@dataclass
class TrainingStats:
    """Training statistics"""
    total_updates: int = 0
    total_experiences: int = 0
    avg_reward: float = 0.0
    avg_position_size: float = 0.5
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "total_experiences": self.total_experiences,
            "avg_reward": round(self.avg_reward, 4),
            "avg_position_size": round(self.avg_position_size, 3),
            "policy_loss": round(self.policy_loss, 6),
            "value_loss": round(self.value_loss, 6),
            "entropy": round(self.entropy, 4),
            "kl_divergence": round(self.kl_divergence, 6),
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent for position sizing.

    State features (8 dimensions):
    1. Regime confidence (0-1)
    2. Volatility regime indicator (normalized)
    3. Signal strength from confidence scorer (0-1)
    4. Recent win rate (rolling window)
    5. Recent avg P&L (normalized)
    6. Current drawdown (0-1)
    7. Position in trend (0-1)
    8. Time of day factor (normalized)

    Action: Position size (0-1)

    Reward: Risk-adjusted return (Sharpe contribution)
    """

    # State feature indices
    IDX_REGIME_CONF = 0
    IDX_VOLATILITY = 1
    IDX_SIGNAL_STRENGTH = 2
    IDX_WIN_RATE = 3
    IDX_AVG_PNL = 4
    IDX_DRAWDOWN = 5
    IDX_TREND_POS = 6
    IDX_TIME_FACTOR = 7
    STATE_SIZE = 8

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[PPOConfig] = None,
    ):
        """
        Initialize the PPO agent.

        Args:
            db_path: Path to database for persistence
            config: PPO hyperparameters
        """
        self.db_path = db_path
        self.config = config or PPOConfig()

        # Policy network (actor)
        self.policy = NetworkWeights()
        self.policy.initialize(self.STATE_SIZE, 16)

        # Value network (critic)
        self.value_net = NetworkWeights()
        self.value_net.initialize(self.STATE_SIZE, 16)

        # Experience buffer
        self.experience_buffer: deque = deque(maxlen=self.config.buffer_size)

        # Training stats
        self.stats = TrainingStats()

        # Rolling performance tracking
        self.recent_rewards: deque = deque(maxlen=100)
        self.recent_positions: deque = deque(maxlen=100)

        # Burn-in flag
        self.is_burned_in = False

        logger.info("PPO Agent initialized")

    def get_state_features(
        self,
        regime_state: Optional[RegimeState] = None,
        signal_strength: float = 0.5,
        recent_performance: Optional[Dict[str, float]] = None,
        current_time: Optional[datetime] = None,
    ) -> List[float]:
        """
        Extract state features from current market conditions.

        Returns 8-dimensional state vector.
        """
        perf = recent_performance or {}

        # Regime confidence
        regime_conf = regime_state.confidence if regime_state else 0.5

        # Volatility indicator (normalized)
        if regime_state:
            vol_indicator = getattr(regime_state, 'atr_percent', 0.02)
            volatility = min(1.0, vol_indicator / 0.05)  # Normalize to 0-1
        else:
            volatility = 0.5

        # Signal strength from confidence scorer
        sig_strength = max(0.0, min(1.0, signal_strength))

        # Recent win rate
        win_rate = perf.get("win_rate", 0.5)

        # Recent avg P&L (normalized to reasonable range)
        avg_pnl = perf.get("avg_pnl_percent", 0.0)
        avg_pnl_norm = max(-1.0, min(1.0, avg_pnl / 5.0))  # ±5% -> ±1

        # Current drawdown
        drawdown = perf.get("current_drawdown", 0.0)
        dd_norm = min(1.0, abs(drawdown) / 20.0)  # 20% max DD -> 1

        # Position in trend (based on regime)
        if regime_state:
            if regime_state.current_regime in [
                MarketRegime.STRONG_UPTREND,
                MarketRegime.BREAKOUT
            ]:
                trend_pos = 0.8
            elif regime_state.current_regime in [
                MarketRegime.WEAK_UPTREND,
            ]:
                trend_pos = 0.6
            elif regime_state.current_regime in [
                MarketRegime.STRONG_DOWNTREND,
                MarketRegime.BREAKDOWN,
            ]:
                trend_pos = 0.2
            elif regime_state.current_regime in [
                MarketRegime.WEAK_DOWNTREND,
            ]:
                trend_pos = 0.4
            else:
                trend_pos = 0.5
        else:
            trend_pos = 0.5

        # Time of day factor (trading session quality)
        if current_time:
            hour = current_time.hour
            # Prefer US market hours (14-21 UTC)
            if 14 <= hour <= 21:
                time_factor = 0.8
            # Asian session (0-7 UTC)
            elif 0 <= hour <= 7:
                time_factor = 0.6
            # European overlap (7-14 UTC)
            elif 7 <= hour <= 14:
                time_factor = 0.7
            else:
                time_factor = 0.5
        else:
            time_factor = 0.5

        return [
            regime_conf,
            volatility,
            sig_strength,
            win_rate,
            avg_pnl_norm,
            dd_norm,
            trend_pos,
            time_factor,
        ]

    def get_action(
        self,
        state: List[float],
        deterministic: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Get position size recommendation.

        Args:
            state: 8-dimensional state vector
            deterministic: If True, return mean (no exploration)

        Returns:
            (action, log_prob, value_estimate)
        """
        # Get policy output
        mean, std = self.policy.forward(state)

        # Get value estimate
        value, _ = self.value_net.forward(state)

        if deterministic or not self.is_burned_in:
            # No exploration during burn-in or inference
            action = mean
        else:
            # Sample from Gaussian
            action = random.gauss(mean, std)

        # Clamp to valid range
        action = max(self.config.min_position_size,
                    min(self.config.max_position_size, action))

        # Calculate log probability
        log_prob = self._log_prob(action, mean, std)

        return action, log_prob, value

    def _log_prob(self, action: float, mean: float, std: float) -> float:
        """Calculate log probability of action under Gaussian policy"""
        var = std ** 2
        log_scale = math.log(std)
        return -((action - mean) ** 2) / (2 * var) - log_scale - 0.5 * math.log(2 * math.pi)

    def store_experience(
        self,
        state: List[float],
        action: float,
        reward: float,
        next_state: List[float],
        done: bool,
        log_prob: float,
        value: float,
        trade_id: Optional[str] = None,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
    ):
        """Store experience in replay buffer"""
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
            trade_id=trade_id,
            symbol=symbol,
            regime=regime,
        )

        self.experience_buffer.append(exp)
        self.recent_rewards.append(reward)
        self.recent_positions.append(action)

        self.stats.total_experiences += 1

        # Update burn-in status
        if not self.is_burned_in and len(self.experience_buffer) >= self.config.min_experiences:
            self.is_burned_in = True
            logger.info(f"PPO Agent burn-in complete with {len(self.experience_buffer)} experiences")

        # Check if update is needed
        if (self.is_burned_in and
            len(self.experience_buffer) >= self.config.batch_size and
            self.stats.total_experiences % self.config.update_frequency == 0):
            self._update()

    def calculate_reward(
        self,
        pnl_percent: float,
        holding_period_hours: float,
        position_size: float,
    ) -> float:
        """
        Calculate risk-adjusted reward for a trade.

        Uses Sharpe-like formulation:
        - Reward = PnL / volatility_contribution
        - Penalty for excessive position sizes
        - Bonus for efficient trades (high return / time)

        Args:
            pnl_percent: Realized P&L percentage
            holding_period_hours: Trade duration
            position_size: Position size used (0-1)

        Returns:
            Scalar reward value
        """
        # Base reward is P&L
        reward = pnl_percent

        # Time efficiency bonus (reward quick wins, penalize slow losses)
        time_factor = 1.0
        if holding_period_hours > 0:
            if pnl_percent > 0:
                # Bonus for quick profits
                time_factor = min(2.0, 24.0 / max(1.0, holding_period_hours))
            else:
                # Penalty for slow losses (should have cut earlier)
                time_factor = max(0.5, 1.0 - (holding_period_hours / 168.0))  # 1 week

        reward *= time_factor

        # Position size risk adjustment
        # Penalize large sizes on losers, reward restraint
        if pnl_percent < 0:
            # Additional penalty for large losing positions
            reward *= (1.0 + position_size)  # Amplifies negative reward
        else:
            # Small bonus for proportional sizing on winners
            reward *= (1.0 + 0.1 * position_size)

        # Normalize to reasonable range
        reward = max(-2.0, min(2.0, reward / 5.0))

        return reward

    def _update(self):
        """Perform PPO update using collected experiences"""
        if len(self.experience_buffer) < self.config.batch_size:
            return

        # Sample batch
        batch = random.sample(list(self.experience_buffer), self.config.batch_size)

        # Compute advantages using GAE
        advantages = self._compute_gae(batch)

        # Multiple epochs of updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.config.n_epochs):
            policy_loss, value_loss, entropy, kl = self._ppo_step(batch, advantages)
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy

            # Early stopping if KL divergence too high
            if kl > 1.5 * self.config.target_kl:
                logger.debug(f"Early stopping due to KL divergence: {kl:.4f}")
                break

        # Update stats
        self.stats.total_updates += 1
        self.stats.policy_loss = total_policy_loss / self.config.n_epochs
        self.stats.value_loss = total_value_loss / self.config.n_epochs
        self.stats.entropy = total_entropy / self.config.n_epochs
        self.stats.kl_divergence = kl
        self.stats.last_update = datetime.now(timezone.utc)

        if self.recent_rewards:
            self.stats.avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        if self.recent_positions:
            self.stats.avg_position_size = sum(self.recent_positions) / len(self.recent_positions)

        logger.info(f"PPO Update #{self.stats.total_updates}: "
                   f"policy_loss={self.stats.policy_loss:.6f}, "
                   f"value_loss={self.stats.value_loss:.6f}, "
                   f"avg_reward={self.stats.avg_reward:.4f}")

        # Persist weights
        self._save_weights()

    def _compute_gae(self, batch: List[Experience]) -> List[float]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        last_gae = 0.0

        # Process in reverse order
        for exp in reversed(batch):
            if exp.done:
                next_value = 0.0
                last_gae = 0.0
            else:
                # Estimate next value
                next_value, _ = self.value_net.forward(exp.next_state)

            delta = exp.reward + self.config.gamma * next_value - exp.value
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages.insert(0, last_gae)

        # Normalize advantages
        mean_adv = sum(advantages) / len(advantages)
        std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages))
        if std_adv > 0:
            advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]

        return advantages

    def _ppo_step(
        self,
        batch: List[Experience],
        advantages: List[float]
    ) -> Tuple[float, float, float, float]:
        """Single PPO update step"""
        policy_loss = 0.0
        value_loss = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0

        for exp, advantage in zip(batch, advantages):
            # Get current policy output
            mean, std = self.policy.forward(exp.state)
            current_log_prob = self._log_prob(exp.action, mean, std)

            # Probability ratio
            ratio = math.exp(current_log_prob - exp.log_prob)

            # Clipped surrogate objective
            clip_adv = max(
                min(ratio, 1.0 + self.config.clip_ratio),
                1.0 - self.config.clip_ratio
            ) * advantage

            policy_loss -= min(ratio * advantage, clip_adv)

            # Value loss
            value_pred, _ = self.value_net.forward(exp.state)
            returns = exp.reward + self.config.gamma * exp.value if not exp.done else exp.reward
            value_loss += (value_pred - returns) ** 2

            # Entropy bonus
            entropy = 0.5 + 0.5 * math.log(2 * math.pi) + math.log(std)
            entropy_sum += entropy

            # KL divergence approximation
            kl = exp.log_prob - current_log_prob
            kl_sum += abs(kl)

        # Average over batch
        n = len(batch)
        policy_loss /= n
        value_loss /= n
        entropy = entropy_sum / n
        kl = kl_sum / n

        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_coef * value_loss -
            self.config.entropy_coef * entropy
        )

        # Simple gradient update (simplified, no backprop)
        # In production, use PyTorch autograd
        self._apply_gradients(batch, advantages, total_loss)

        return policy_loss, value_loss, entropy, kl

    def _apply_gradients(
        self,
        batch: List[Experience],
        advantages: List[float],
        loss: float
    ):
        """Apply gradient update (simplified finite difference)"""
        # This is a simplified update for demo purposes
        # In production, use proper autodiff (PyTorch/JAX)

        lr = self.config.learning_rate

        # Update policy weights based on advantage-weighted gradients
        for exp, adv in zip(batch, advantages):
            # Small perturbation in direction of advantage
            for i in range(len(self.policy.b1)):
                for j in range(len(self.policy.w1)):
                    if j < len(exp.state):
                        grad = exp.state[j] * adv * 0.01
                        self.policy.w1[j][i] += lr * grad

            # Update log_std towards lower variance on good outcomes
            if adv > 0:
                self.policy.log_std -= lr * 0.01
            else:
                self.policy.log_std += lr * 0.001

            # Clamp log_std
            self.policy.log_std = max(-2.0, min(0.5, self.policy.log_std))

    def recommend_position_size(
        self,
        regime_state: Optional[RegimeState] = None,
        signal_strength: float = 0.5,
        recent_performance: Optional[Dict[str, float]] = None,
        confidence_score: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Get position size recommendation with explanation.

        Returns:
            Dict with size, confidence, factors, and raw values
        """
        # Get state features
        state = self.get_state_features(
            regime_state=regime_state,
            signal_strength=signal_strength,
            recent_performance=recent_performance,
        )

        # Get action from policy
        action, log_prob, value = self.get_action(state, deterministic=True)

        # Blend with confidence score
        # PPO output weighted more after burn-in
        if self.is_burned_in:
            ppo_weight = 0.6
            confidence_weight = 0.4
        else:
            ppo_weight = 0.2
            confidence_weight = 0.8

        blended_size = action * ppo_weight + confidence_score * confidence_weight
        blended_size = max(self.config.min_position_size,
                          min(self.config.max_position_size, blended_size))

        # Identify key factors
        factors = []
        if state[self.IDX_DRAWDOWN] > 0.5:
            factors.append("High drawdown - reducing size")
            blended_size *= 0.7
        if state[self.IDX_WIN_RATE] > 0.6:
            factors.append("Strong recent performance")
        if state[self.IDX_VOLATILITY] > 0.7:
            factors.append("High volatility - cautious sizing")
            blended_size *= 0.8
        if state[self.IDX_SIGNAL_STRENGTH] > 0.7:
            factors.append("Strong signal quality")

        return {
            "recommended_size": round(blended_size, 3),
            "ppo_raw": round(action, 3),
            "confidence_component": round(confidence_score, 3),
            "ppo_weight": ppo_weight,
            "value_estimate": round(value, 4),
            "is_burned_in": self.is_burned_in,
            "experiences": self.stats.total_experiences,
            "factors": factors,
            "state_summary": {
                "regime_confidence": round(state[0], 2),
                "volatility": round(state[1], 2),
                "signal_strength": round(state[2], 2),
                "win_rate": round(state[3], 2),
                "drawdown": round(state[5], 2),
            }
        }

    def _save_weights(self):
        """Persist weights to database"""
        if not self.db_path:
            return

        try:
            conn = sqlite3.connect(self.db_path)

            # Ensure table exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ppo_weights (
                    weight_id TEXT PRIMARY KEY,
                    policy_weights TEXT NOT NULL,
                    value_weights TEXT NOT NULL,
                    stats TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            weight_id = "current"
            policy_json = json.dumps(self.policy.to_dict())
            value_json = json.dumps(self.value_net.to_dict())
            stats_json = json.dumps(self.stats.to_dict())

            conn.execute("""
                INSERT OR REPLACE INTO ppo_weights
                (weight_id, policy_weights, value_weights, stats, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (weight_id, policy_json, value_json, stats_json,
                 datetime.now(timezone.utc).isoformat()))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to save PPO weights: {e}")

    def load_weights(self) -> bool:
        """Load weights from database"""
        if not self.db_path:
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT policy_weights, value_weights, stats
                FROM ppo_weights WHERE weight_id = 'current'
            """)
            row = cursor.fetchone()
            conn.close()

            if row:
                self.policy = NetworkWeights.from_dict(json.loads(row[0]))
                self.value_net = NetworkWeights.from_dict(json.loads(row[1]))

                stats_data = json.loads(row[2])
                self.stats.total_updates = stats_data.get("total_updates", 0)
                self.stats.total_experiences = stats_data.get("total_experiences", 0)
                self.stats.avg_reward = stats_data.get("avg_reward", 0.0)

                # Check burn-in status
                if self.stats.total_experiences >= self.config.min_experiences:
                    self.is_burned_in = True

                logger.info(f"Loaded PPO weights: {self.stats.total_updates} updates, "
                           f"{self.stats.total_experiences} experiences")
                return True

        except Exception as e:
            logger.warning(f"Failed to load PPO weights: {e}")

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get agent status summary"""
        return {
            "is_burned_in": self.is_burned_in,
            "buffer_size": len(self.experience_buffer),
            "min_for_burnin": self.config.min_experiences,
            "stats": self.stats.to_dict(),
            "config": {
                "learning_rate": self.config.learning_rate,
                "clip_ratio": self.config.clip_ratio,
                "update_frequency": self.config.update_frequency,
            }
        }


# Convenience function
def create_ppo_agent(db_path: Optional[str] = None, **kwargs) -> PPOAgent:
    """Create and optionally load a PPO agent"""
    config = PPOConfig(**kwargs)
    agent = PPOAgent(db_path=db_path, config=config)

    if db_path:
        agent.load_weights()

    return agent
