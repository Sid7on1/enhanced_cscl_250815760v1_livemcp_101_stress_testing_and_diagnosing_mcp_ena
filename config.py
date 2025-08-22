import os
import logging
from typing import Dict, List
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class
class Config(ABC):
    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.validate_config()

    @abstractmethod
    def validate_config(self):
        """Validate the configuration and raise errors for missing/incorrect fields."""
        pass

    @abstractmethod
    def update_config(self, updates: Dict):
        """Update the configuration with new values."""
        pass

    @property
    def device(self) -> torch.device:
        """Return the device to use for tensor operations."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agent configuration
class AgentConfig(Config):
    def __init__(self, config_dict: Dict):
        super().__init__(config_dict)
        self.model_path = self.config.get('model_path')
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.batch_size = self.config.get('batch_size', 32)
        self.num_epochs = self.config.get('num_epochs', 10)
        self.input_dim = self.config.get('input_dim', 32)
        self.output_dim = self.config.get('output_dim', 10)
        self._validate_agent_config()

    def _validate_agent_config(self):
        """Validate agent-specific configuration values."""
        if not self.model_path:
            raise ValueError("Model path not specified in configuration.")
        if not os.path.isfile(self.model_path):
            raise ValueError(f"Model path '{self.model_path}' does not exist.")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be a positive value.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if self.num_epochs < 0:
            raise ValueError("Number of epochs must be a non-negative integer.")
        if self.input_dim < 0 or self.output_dim < 0:
            raise ValueError("Input and output dimensions must be non-negative integers.")

    def update_config(self, updates: Dict):
        super().update_config(updates)
        if 'model_path' in updates:
            self.model_path = updates['model_path']
        if 'learning_rate' in updates:
            self.learning_rate = updates['learning_rate']
        if 'batch_size' in updates:
            self.batch_size = updates['batch_size']
        if 'num_epochs' in updates:
            self.num_epochs = updates['num_epochs']
        if 'input_dim' in updates:
            self.input_dim = updates['input_dim']
        if 'output_dim' in updates':
            self.output_وتوزع_dim = updates['output_dim']

    @property
    def model_path(self) -> str:
        """Return the path to the trained model."""
        return self._model_path

    @model_path.setter
    def model_path(self, path: str):
        """Set the path to the trained model."""
        self._model_path = path

# Environment configuration
class EnvConfig(Config):
    def __init__(self, config_dict: Dict):
        super().__init__(config_dict)
        self.env_name = self.config.get('env_name')
        self.num_episodes = self.config.get('num_episodes', 100)
        self.max_steps = self.config.get('max_steps', 1000)
        self.reward_threshold = self.config.get('reward_threshold', 100.0)
        self._validate_env_config()

    def _validate_env_config(self):
        """Validate environment-specific configuration values."""
        if not self.env_name:
            raise ValueError("Environment name not specified in configuration.")
        if self.num_episodes < 0:
            raise ValueError("Number of episodes must be a non-negative integer.")
        if self.max_steps < 0:
            raise ValueError("Maximum steps per episode must be a non-negative integer.")
        if self.reward_threshold < 0:
            raise ValueError("Reward threshold must be a non-negative value.")

    def update_config(self, updates: Dict):
        super().update_config(updates)
        if 'env_name' in updates:
            self.env_name = updates['env_name']
        if 'num_episodes' in updates:
            self.num_episodes = updates['num_episodes']
        if 'max_steps' in updates:
            self.max_steps = updates['max_steps']
        if 'reward_threshold' in updates:
            self.reward_threshold = updates['reward_threshold']

# Project-specific configuration
class ProjectConfig:
    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.agent_config = self.config.get('agent', {})
        self.env_config = self.config.get('environment', {})
        self.data_path = self.config.get('data_path')
        self.log_dir = self.config.get('log_dir', 'logs')
        self.seed = self.config.get('random_seed', 42)
        self._validate_project_config()

    def _validate_project_config(self):
        """Validate project-specific configuration values."""
        if not self.agent_config:
            raise ValueError("Agent configuration not found in project config.")
        if not self.env_config:
            raise ValueError("Environment configuration not found in project config.")
        if self.data_path and not os.path.isdir(self.data_path):
            raise ValueError(f"Data path '{self.data_path}' is not a valid directory.")
        if not os.path.isdir(self.log_dir):
            raise ValueError(f"Log directory '{self.log_dir}' does not exist.")
        if self.seed is not None and (not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError("Random seed must be a non-negative integer or None.")

    @property
    def agent_config(self) -> Dict:
        """Return the agent-specific configuration."""
        return self._agent_config

    @agent_config.setter
    def agent_config(self, config: Dict):
        """Set the agent-specific configuration."""
        self._validate_agent_config(config)
        self._agent_config = AgentConfig(config)

    def _validate_agent_config(self, config: Dict):
        """Validate agent configuration values."""
        required_fields = ['model_path', 'learning_rate', 'batch_size', 'num_epochs']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in agent config.")

    @property
    def env_config(self) -> Dict:
        """Return the environment-specific configuration."""
        return self._env_config

    @env_config.setter
    def env_config(self, config: Dict):
        """Set the environment-specific configuration."""
        self._validate_env_config(config)
        self._env_config = EnvConfig(config)

    def _validate_env_config(self, config: Dict):
        """Validate environment configuration values."""
        required_fields = ['env_name', 'num_episodes', 'max_steps', 'reward_threshold']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in environment config.")

# Function to load configuration from a file
def load_config(config_file: str) -> ProjectConfig:
    """Load the project configuration from a file."""
    if not os.path.isfile(config_file):
        raise ValueError(f"Configuration file '{config_file}' not found.")
    with open(config_file, 'r') as file:
        config_data = file.read()
        config_dict = json.loads(config_data)
        return ProjectConfig(config_dict)

# Function to update configuration with command-line arguments
def update_config_from_cli(config: ProjectConfig, args: List[str]) -> ProjectConfig:
    """Update the project configuration based on command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_model", help="Path to the trained agent model.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--env_name", help="Name of the environment to use.")
    parser.add_argument("--num_episodes", type=int, help="Number of episodes to run.")
    parser.add_argument("--max_steps", type=int, help="Maximum steps per episode.")
    parser.add_argument("--reward_threshold", type=float, help="Reward threshold for success.")
    parser.add_argument("--data_path", help="Path to the data directory.")
    parser.add_argument("--log_dir", help="Directory for log files.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    args = parser.parse_args(args)

    # Update agent configuration
    if args.agent_model:
        config.agent_config.model_path = args.agent_model
    if args.learning_rate:
        config.agent_config.learning_rate = args.learning_rate
    if args.batch_size:
        config.agent_config.batch_size = args.batch_size
    if args.num_epochs:
        config.agent_config.num_epochs = args.num_epochs

    # Update environment configuration
    if args.env_name:
        config.env_config.env_name = args.env_name
    if args.num_episodes:
        config.env_config.num_episodes = args.num_episodes
    if args.max_steps:
        config.env_config.max_steps = args.max_steps
    if args.reward_threshold:
        config.env_config.reward_threshold = args.reward_threshold

    # Update general configuration
    if args.data_path:
        config.data_path = args.data_path
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.seed:
        config.seed = args.seed

    return config

# Example usage
if __name__ == "__main__":
    import argparse
    config_file = "path/to/config.json"
    project_config = load_config(config_file)

    # Update configuration with command-line arguments (if any)
    updated_config = update_config_from_cli(project_config, sys.argv[1:])

    # Access configuration values
    agent_config = updated_config.agent_config
    env_config = updated_config.env_config
    data_path = updated_config.data_path
    log_dir = updated_config.log_dir
    random_seed = updated_config.seed

    # Perform additional setup and run the agent
    ...