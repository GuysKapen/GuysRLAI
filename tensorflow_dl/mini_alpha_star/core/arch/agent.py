import numpy as np

import tensorflow as tf

from pysc2.lib import actions as A

from tensorflow_dl.mini_alpha_star.core.arch.arch_model import ArchModel
from tensorflow_dl.mini_alpha_star.core.arch.entity_encoder import Entity
from tensorflow_dl.mini_alpha_star.core.rl.action import ArgsAction
from tensorflow_dl.mini_alpha_star.core.rl.state import MsState
from tensorflow_dl.mini_alpha_star.core.sl.feature import Feature
from tensorflow_dl.mini_alpha_star.core.sl.label import Label
from tensorflow_dl.mini_alpha_star.libs import utils as L
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import MiniStar_Arch_Hyper_Parameters as MAHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS

from pysc2.lib.units import get_unit_type


class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()