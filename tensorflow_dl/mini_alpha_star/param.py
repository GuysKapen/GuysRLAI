from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

"""whether is running on server, on server meaning use GPU with larger memoary"""
on_server = False
# on_server = True

'''The replay path'''
replay_path = "data/Replays/filtered_replays_1/"
# replay_path = "/home/liuruoze/data4/mini-AlphaStar/data/filtered_replays_1/"
# replay_path = "/home/liuruoze/mini-AlphaStar/data/filtered_replays_1/"

'''The mini scale used in hyperparameter'''
# Mini_Scale = 4
# Mini_Scale = 8
Mini_Scale = 16

debug = True

# Minimap index
_M_HEIGHT = features.MINIMAP_FEATURES.height_map.index
_M_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_M_CAMERA = features.MINIMAP_FEATURES.camera.index
_M_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_M_SELECTED = features.MINIMAP_FEATURES.selected.index