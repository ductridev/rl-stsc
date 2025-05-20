from src.memory_nstep import NStepReplayMemory
from src.utils import set_train_path, set_sumo
from src.per import PERMemory
from src.model import DQN
from src.utils import import_train_configuration

config = import_train_configuration('config/training_cfg.ini')

memory = NStepReplayMemory(capacity=100_000, n_step=3, gamma=0.99)