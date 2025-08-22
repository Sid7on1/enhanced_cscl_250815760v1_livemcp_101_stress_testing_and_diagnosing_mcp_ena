import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from scipy.spatial import distance
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MEMORY_SIZE = 10000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99

# Enum for memory types
class MemoryType(Enum):
    EXPERIENCE = 1
    TRANSITION = 2

# Dataclass for experience
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Dataclass for transition
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Memory class
class Memory(ABC):
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)

    @abstractmethod
    def add(self, experience: Experience):
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Experience]:
        pass

# Experience replay memory
class ExperienceReplayMemory(Memory):
    def __init__(self, memory_size: int):
        super().__init__(memory_size)

    def add(self, experience: Experience):
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return np.random.choice(self.memory, batch_size, replace=False).tolist()

# Transition memory
class TransitionMemory(Memory):
    def __init__(self, memory_size: int):
        super().__init__(memory_size)

    def add(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return np.random.choice(self.memory, batch_size, replace=False).tolist()

# Experience replay buffer
class ExperienceReplayBuffer:
    def __init__(self, memory_size: int):
        self.memory = ExperienceReplayMemory(memory_size)

    def add(self, experience: Experience):
        self.memory.add(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return self.memory.sample(batch_size)

# Transition buffer
class TransitionBuffer:
    def __init__(self, memory_size: int):
        self.memory = TransitionMemory(memory_size)

    def add(self, transition: Transition):
        self.memory.add(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return self.memory.sample(batch_size)

# Experience replay buffer with transition buffer
class ExperienceReplayBufferWithTransitionBuffer:
    def __init__(self, memory_size: int):
        self.experience_replay_buffer = ExperienceReplayBuffer(memory_size)
        self.transition_buffer = TransitionBuffer(memory_size)

    def add_experience(self, experience: Experience):
        self.experience_replay_buffer.add(experience)

    def add_transition(self, transition: Transition):
        self.transition_buffer.add(transition)

    def sample_experience(self, batch_size: int) -> List[Experience]:
        return self.experience_replay_buffer.sample(batch_size)

    def sample_transition(self, batch_size: int) -> List[Transition]:
        return self.transition_buffer.sample(batch_size)

# Main class
class MemoryManager:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.experience_replay_buffer = ExperienceReplayBuffer(memory_size)
        self.transition_buffer = TransitionBuffer(memory_size)

    def add_experience(self, experience: Experience):
        self.experience_replay_buffer.add_experience(experience)

    def add_transition(self, transition: Transition):
        self.transition_buffer.add_transition(transition)

    def sample_experience(self, batch_size: int) -> List[Experience]:
        return self.experience_replay_buffer.sample_experience(batch_size)

    def sample_transition(self, batch_size: int) -> List[Transition]:
        return self.transition_buffer.sample_transition(batch_size)

# Usage
if __name__ == "__main__":
    memory_manager = MemoryManager(MEMORY_SIZE)

    # Create experiences
    experiences = []
    for i in range(10):
        experience = Experience(np.random.rand(4), np.random.randint(0, 5), np.random.rand(), np.random.rand(4), False)
        experiences.append(experience)

    # Add experiences to memory
    for experience in experiences:
        memory_manager.add_experience(experience)

    # Sample experiences
    batch_size = 5
    sampled_experiences = memory_manager.sample_experience(batch_size)
    for experience in sampled_experiences:
        print(experience)

    # Create transitions
    transitions = []
    for i in range(10):
        transition = Transition(np.random.rand(4), np.random.randint(0, 5), np.random.rand(), np.random.rand(4), False)
        transitions.append(transition)

    # Add transitions to memory
    for transition in transitions:
        memory_manager.add_transition(transition)

    # Sample transitions
    batch_size = 5
    sampled_transitions = memory_manager.sample_transition(batch_size)
    for transition in sampled_transitions:
        print(transition)