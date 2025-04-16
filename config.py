# Hyperparameters and configuration for SAC experiments
import torch

# Environment
ENV_NAME = "LunarLanderContinuous-v2"
SEED = 42
MAX_STEPS = 1000000
EVAL_FREQUENCY = 5000
NUM_EVAL_EPISODES = 10

# SAC hyperparameters
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
AUTO_ENTROPY_TUNING = True
TARGET_ENTROPY = None  # Will be set automatically based on action space

# Network architecture
ACTOR_HIDDEN_DIMS = [256, 256]
CRITIC_HIDDEN_DIMS = [256, 256]
QUANTUM_LAYERS = 2  # Number of variational layers for quantum circuit
QUANTUM_WIRES = 4   # Number of qubits

# Training
BATCH_SIZE = 256
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4
REPLAY_BUFFER_SIZE = 1000000
INITIAL_RANDOM_STEPS = 10000
GRADIENT_STEPS = 1
UPDATE_AFTER = 1000

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")