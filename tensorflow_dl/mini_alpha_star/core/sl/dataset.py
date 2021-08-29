import os
import time
import traceback

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_dl.mini_alpha_star.libs.hyper_params import DATASET_SPLIT_RATIO


class SC2ReplayData(object):
    def __init__(self):
        self.initialed = False
        self.feature_size = None
        self.label_size = None
        self.replay_length_list = []

    def get_trainable_data(self, path):
        out_features = None
        print(f"Data path: {path}")
        replay_files = os.listdir(path)

        print(f"Length of replay_files: {replay_files}")
        replay_files.sort()

        traj_list = []
        for i, replay_file in enumerate(replay_files):
            try:
                if i > 15:
                    break
                replay_path = path + replay_file
                print(f"Replay path {replay_path}")

                m = tfds.load(replay_path)
                features = m['features']
                labels = m['labels']
                assert len(features) == len(labels)

                if self.feature_size:
                    assert self.feature_size == features.shape[1]
                    assert self.label_size == labels.shape[1]
                else:
                    self.feature_size = features.shape[1]
                    self.label_size = labels.shape[1]

                print(f"feature_size: {self.feature_size}")
                print(f"label_size {self.label_size}")

                is_final = tf.zeros([features.shape[0], 1])
                is_final[features.shape[0] - 1, 0] = 1

                one_traj = tf.concat([features, labels, is_final], dim=1)

                traj_list.append(one_traj)
                self.replay_length_list.append(one_traj.shape[0])
            except Exception as e:
                traceback.print_exc()

        print("End")
        print(f"self.replay_length_list: {self.replay_length_list}")

        self.initialed = True
        return traj_list

    @staticmethod
    def filter_data(feature, label):
        tmp_feature = None
        return tmp_feature

    @staticmethod
    def get_training_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        print(f"training_size: {training_size}")
        return trajs[0:training_size]

    @staticmethod
    def get_val_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        val_size = int(len(trajs) * DATASET_SPLIT_RATIO.val)
        print(f"val_size: {val_size}")
        return trajs[training_size:training_size + val_size]

    @staticmethod
    def get_test_data(traj):
        test_size = int(len(traj) * DATASET_SPLIT_RATIO.test)
        print(f"test_size: {test_size}")
        return traj[-test_size:]

    @staticmethod
    def get_training_for_val_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        print(f'training_size {training_size}')
        return trajs[0:training_size]

    @staticmethod
    def get_training_for_test_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        val_size = int(len(trajs) * DATASET_SPLIT_RATIO.val)
        print('training_for_test_size:', training_size + val_size)
        return trajs[0:training_size + val_size]

    @staticmethod
    def get_training_for_deploy_data(trajs):
        training_size = int(len(trajs) * DATASET_SPLIT_RATIO.training)
        val_size = int(len(trajs) * DATASET_SPLIT_RATIO.val)
        test_size = int(len(trajs) * DATASET_SPLIT_RATIO.test)
        print('training_for_deploy_size:', training_size + val_size + test_size)
        return trajs[0:training_size + val_size + test_size]


class SC2ReplayDataset(object):
    def __init__(self, traj_list, seq_length, training=False):
        super(SC2ReplayDataset, self).__init__()

        self.traj_list = traj_list
        self.seq_length = seq_length

    def __getitem__(self, index):
        old_start = 0
        begin = index * self.seq_length
        end = (index + 1) * self.seq_length

        for i, one_traj in enumerate(self.traj_list):
            new_start = old_start + one_traj.shape[0]
            if begin >= new_start:
                pass
            else:
                index_begin = begin - old_start
                if end < new_start:
                    index_end = end - old_start
                    return one_traj[index_begin:index_end, :]
                elif i < len(self.traj_list) - 1:
                    next_traj = self.traj_list[i + 1]

                    first_part = one_traj[index_begin:, :]
                    second_part = next_traj[:self.seq_length - len(first_part), :]
                    return tf.concat([first_part, second_part], axis=0)

            old_start = new_start

    def __len__(self):
        max_len = 0
        for one_traj in self.traj_list:
            max_len += one_traj.shape[0]
        max_len -= self.seq_length

        return int(max_len / self.seq_length)

