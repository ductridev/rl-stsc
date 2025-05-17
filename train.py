from src.memory_nstep import NStepReplayMemory
from src.utils import set_train_path, set_sumo
from src.per import PERMemory
from src.model import DQN

memory = NStepReplayMemory(capacity=100_000, n_step=3, gamma=0.99)