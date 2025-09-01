import torch

# Data paths
DATA_PATH = "/dms/workspace_2025/vuthede/VLM/RL-VLM/notebooks/circles_dataset"
TRAIN_DATASET_PATH = f"{DATA_PATH}/train"
VAL_DATASET_PATH = f"{DATA_PATH}/val"
TEST_DATASET_PATH = f"{DATA_PATH}/test"
TEST_HARD_NUMBER_DATASET_PATH = f"{DATA_PATH}/test_hard"

# JSON data files (if used)
TRAIN_JSON = "label.json"
VAL_JSON = "label.json"
TEST_JSON = "label.json"

# Model settings
MODEL_NAME = "microsoft/Florence-2-base-ft"
MODEL_REVISION = "refs/pr/6"
MODEL_VISION_TOWER_TRAIN = False
TRUST_REMOTE_CODE = True

# Training parameters
TRAIN_SUBSET_SIZE = 5000
TEST_SUBSET_SIZE = 1000
TEST_HARD_SUBSET_SIZE = 1000
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 4
LEARNING_RATE = 1e-5


# Device settings
USE_ACCELERATOR = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else DEVICE)

# GRPO training parameters
POLICY_UPDATE="GRPO"
# POLICY_UPDATE="PPO"

GRPO_EPOCHS = 1
# GROUP_SIZE = 3 # To small
GROUP_SIZE = 8
CLIP_COEF = 0.1
ENTROPY_COEF = 0.001
# ENTROPY_COEF = 0.01 # For more exploration. Set this one might not good, just move to the old one

UPDATE_EPOCHS = 2
ROLLOUT_BATCH_SIZE = 1 # Currenly in code use the Batch_size
# MINIBATCH_SIZE = 4 # to small can affect gradients
MINIBATCH_SIZE = 16

# MAX_NEW_TOKENS = 120
MAX_NEW_TOKENS = 150

ITERATIONS_PER_EPOCH = 50
GRPO_ITERATIONS = 20
# GRPO_LEARNING_RATE = 4e-8
# GRPO_LEARNING_RATE = 4e-6
GRPO_LEARNING_RATE = 1e-6
# GRPO_LEARNING_RATE = 1e-5


# Logs
TENSORBOARD_LOG = True 