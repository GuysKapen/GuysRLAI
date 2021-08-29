import os

USED_DEVICES = "0"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import time
import traceback
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from absl import flags
from absl import app
from tqdm import tqdm

# from tensorflow_dl.mini_alpha_star.core.arch.agent import Agent
from tensorflow_dl.mini_alpha_star.core.sl.feature import Feature
from tensorflow_dl.mini_alpha_star.core.sl.label import Label
from tensorflow_dl.mini_alpha_star.core.sl import losses as Loss
# from tensorflow_dl.mini_alpha_star.core.sl.dataset import OneReplayDataset, AllReplayDataset
from tensorflow_dl.mini_alpha_star.libs.utils import load_latest_model
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import SL_Training_Hyper_parameters as SLTHP

__author__ = "Ruo-Ze Liu"

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="./data/replay_data/", help="The path where data stored")
parser.add_argument("-r", "--restore", action="store_true", default=False, help="whether to restore model or not")
parser.add_argument("-t", "--type", choices=["val", "test", "deploy"], default="val", help="Train type")
parser.add_argument("-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type")
parser.add_argument("-n", "--norm", choices=[True, False], default=False, help="Use norm for data")
args = parser.parse_args()

# training paramerters
PATH = args.path
MODEL = args.model
TYPE = args.type
RESTORE = args.restore
NORM = args.norm

# hyper paramerters
BATCH_SIZE = AHP.batch_size
print('BATCH_SIZE:', BATCH_SIZE) if debug else None
SEQ_LEN = AHP.sequence_length
print('SEQ_LEN:', SEQ_LEN) if debug else None

NUM_EPOCHS = 100  # SLTHP.num_epochs
LEARNING_RATE = 1e-3  # SLTHP.learning_rate
WEIGHT_DECAY = 1e-5  # SLTHP.weight_decay
CLIP = 0.5  # SLTHP.clip

NUM_ITERS = 100  # 100
FILE_SIZE = 100  # 100

# set random seed
tf.random.set_seed(SLTHP.seed)
np.random.seed(SLTHP.seed)

# model path
MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time.strftime("%y-%m-%d_%H-%M-%S", time.localtime()))
