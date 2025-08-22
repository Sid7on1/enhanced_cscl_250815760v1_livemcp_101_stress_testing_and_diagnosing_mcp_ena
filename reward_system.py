import logging
import numpy as np
from typing import Dict, List, Tuple
from reward_system.config import Config
from reward_system.exceptions import RewardSystemError
from reward_system.models import RewardModel
from reward_system.utils import calculate_velocity, calculate_flow

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class is responsible for calculating rewards based on the agent's actions
    and the environment's state. It uses the velocity-threshold and Flow Theory
    algorithms to calculate rewards.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reward_model = RewardModel(config)

    def calculate_reward(self, action: np.ndarray, state: np.ndarray) -> float:
        """
        Calculate the reward for the given action and state.

        Args:
            action: Action taken by the agent.
            state: Current state of the environment.

        Returns:
            Reward value.
        """
        try:
            velocity = calculate_velocity(action, state)
            flow = calculate_flow(action, state)
            reward = self.reward_model.calculate_reward(velocity, flow)
            return reward
        except RewardSystemError as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward to make it more suitable for the agent's learning process.

        Args:
            reward: Reward value.

        Returns:
            Shaped reward value.
        """
        if reward > self.config.max_reward:
            return self.config.max_reward
        elif reward < self.config.min_reward:
            return self.config.min_reward
        else:
            return reward

class RewardModel:
    """
    Reward model.

    This class is responsible for calculating rewards based on the velocity and flow
    values.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward model.

        Args:
            config: Configuration object.
        """
        self.config = config

    def calculate_reward(self, velocity: float, flow: float) -> float:
        """
        Calculate the reward based on the velocity and flow values.

        Args:
            velocity: Velocity value.
            flow: Flow value.

        Returns:
            Reward value.
        """
        if velocity > self.config.velocity_threshold:
            return self.config.flow_reward
        elif flow > self.config.flow_threshold:
            return self.config.velocity_reward
        else:
            return 0.0

class Config:
    """
    Configuration object.

    This class is responsible for storing configuration settings.
    """

    def __init__(self):
        """
        Initialize the configuration object.
        """
        self.max_reward = 1.0
        self.min_reward = -1.0
        self.velocity_threshold = 0.5
        self.flow_threshold = 0.5
        self.velocity_reward = 0.5
        self.flow_reward = 0.5

class RewardSystemError(Exception):
    """
    Reward system error.

    This exception is raised when an error occurs in the reward system.
    """

    def __init__(self, message: str):
        """
        Initialize the exception.

        Args:
            message: Error message.
        """
        self.message = message

def calculate_velocity(action: np.ndarray, state: np.ndarray) -> float:
    """
    Calculate the velocity value.

    Args:
        action: Action taken by the agent.
        state: Current state of the environment.

    Returns:
        Velocity value.
    """
    # Calculate velocity using the formula from the paper
    velocity = np.linalg.norm(action - state)
    return velocity

def calculate_flow(action: np.ndarray, state: np.ndarray) -> float:
    """
    Calculate the flow value.

    Args:
        action: Action taken by the agent.
        state: Current state of the environment.

    Returns:
        Flow value.
    """
    # Calculate flow using the formula from the paper
    flow = np.dot(action, state)
    return flow

if __name__ == "__main__":
    # Create a configuration object
    config = Config()

    # Create a reward system object
    reward_system = RewardSystem(config)

    # Create an action and state array
    action = np.array([1.0, 2.0, 3.0])
    state = np.array([4.0, 5.0, 6.0])

    # Calculate the reward
    reward = reward_system.calculate_reward(action, state)

    # Shape the reward
    shaped_reward = reward_system.shape_reward(reward)

    # Print the reward
    print(f"Reward: {reward}")
    print(f"Shaped Reward: {shaped_reward}")