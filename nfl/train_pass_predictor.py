import numpy as np
import torch

from core.trainer.pipeline import PipelineDriver
from core.trainer.solver import Solver
from core.utils.lib.data_handling import create_train_test_split
from nfl.data_management.data_manager import DataManager
from nfl.lib.constants import EXCLUDED_PLAY_TYPES, PLAY_COLS, SCENARIO_COLS

def _get_parameter_grid(input_size: int = 10):
    MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS = 1, 20
    MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE = input_size, input_size*30
    MIN_LEARNING_RATE, MAX_LEARNING_RATE = 1e-5, 1e-2
    MIN_BATCH_SIZE, MAX_BATCH_SIZE = 64, 512
    MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY = 1e-5, 1e-2
    MIN_DROPOUT_RATE, MAX_DROPOUT_RATE = 0.0, 0.5
    MIN_RELU_SLOPE, MAX_RELU_SLOPE = 0.01, 0.1

    param_space = {
        "batch_size": (MIN_BATCH_SIZE, MAX_BATCH_SIZE, "int"),
        "lr": (MIN_LEARNING_RATE, MAX_LEARNING_RATE, "log"),
        "num_hidden_layers": (MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS, "int"),
        "hidden_size": (MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, "int"),
        "weight_decay": (MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, "log"),
        "dropout_rate": (MIN_DROPOUT_RATE, MAX_DROPOUT_RATE, "float"),
        "relu_slope": (MIN_RELU_SLOPE, MAX_RELU_SLOPE, "float")
    }

    return param_space

def train_pass_outcome_predictor(num_epoch: int = 100, num_splits: int = 5, data = None, feature_cols = None):