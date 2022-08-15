import os

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from data_calibration import calibrate_amplitude_custom, calibrate_phase, calibrate_amplitude, dwn_noise, hampel

DATASET_FOLDER = "../dataset"
DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

SUBCARRIES_NUM_TWO_HHZ = 56
SUBCARRIES_NUM_FIVE_HHZ = 114

PHASE_MIN, PHASE_MAX = 3.1389, 3.1415
AMP_MIN, AMP_MAX = 0.0, 577.6582


def read_csi_data_from_csv(path_to_csv, is_five_hhz=False, antenna_pairs=4):
    """
    Read csi data(amplitude, phase) from .csv data

    :param path_to_csv: string
    :param is_five_hhz: boolean
    :param antenna_pairs: integer
    :return: (amplitudes, phases) => (np.array of shape(data len, num_of_subcarriers * antenna_pairs),
                                     np.array of shape(data len, num_of_subcarriers * antenna_pairs))
    """

    data = pd.read_csv(path_to_csv, header=None).values

    if is_five_hhz:
        subcarries_num = SUBCARRIES_NUM_FIVE_HHZ
    else:
        subcarries_num = SUBCARRIES_NUM_TWO_HHZ

    # 1 -> to skip subcarriers numbers in data
    amplitudes = data[:, subcarries_num * 1:subcarries_num * (1 + antenna_pairs)]
    phases = data[:, subcarries_num * (1 + antenna_pairs):subcarries_num * (1 + 2 * antenna_pairs)]

    return amplitudes, phases


def read_labels_from_csv(path_to_csv):
    """
    Read labels(human activities) from csv file

    :param path_to_csv: string
    :return: labels, np.array of shape(data_len, 1)
    """

    data = pd.read_csv(path_to_csv, header=None).values
    labels = data[:, 1]

    return labels


def read_all_data_from_files(paths, is_five_hhz=True, antenna_pairs=4):
    """
    Read csi and labels data from all folders in the dataset

    :return: amplitudes, phases, labels all of shape (data len, num of subcarriers)
    """

    final_amplitudes, final_phases, final_labels = np.empty((0, antenna_pairs * SUBCARRIES_NUM_FIVE_HHZ)), \
                                                   np.empty((0, antenna_pairs * SUBCARRIES_NUM_FIVE_HHZ)), \
                                                   np.empty((0))

    for index, path in enumerate(paths):
        amplitudes, phases = read_csi_data_from_csv(os.path.join(path, "data.csv"), is_five_hhz, antenna_pairs)
        labels = read_labels_from_csv(os.path.join(path, "label.csv"))

        amplitudes, phases = amplitudes[:-1], phases[:-1]  # fix the bug with the last element

        final_amplitudes = np.concatenate((final_amplitudes, amplitudes))
        final_phases = np.concatenate((final_phases, phases))
        final_labels = np.concatenate((final_labels, labels))

    return final_amplitudes, final_phases, final_labels


def read_all_data(is_five_hhz=True, antenna_pairs=4):
    all_paths = []

    for index, room in enumerate(DATA_ROOMS):
        for subroom in DATA_SUBROOMS[index]:
            all_paths.append(os.path.join(DATASET_FOLDER, room, subroom))

    return read_all_data_from_files(all_paths, is_five_hhz, antenna_pairs)


class CSIDataset(Dataset):
    """CSI Dataset."""

    def __init__(self, csv_files, window_size=32, step=1):
        from sklearn import decomposition

        self.amplitudes, self.phases, self.labels = read_all_data_from_files(csv_files)

        self.amplitudes = calibrate_amplitude(self.amplitudes)

        pca = decomposition.PCA(n_components=3)

        # self.phases[:, 0 * SUBCARRIES_NUM_FIVE_HHZ:1 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 0 * SUBCARRIES_NUM_FIVE_HHZ:1 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 1 * SUBCARRIES_NUM_FIVE_HHZ:2 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 1 * SUBCARRIES_NUM_FIVE_HHZ:2 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 2 * SUBCARRIES_NUM_FIVE_HHZ:3 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 2 * SUBCARRIES_NUM_FIVE_HHZ:3 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 3 * SUBCARRIES_NUM_FIVE_HHZ:4 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 3 * SUBCARRIES_NUM_FIVE_HHZ:4 * SUBCARRIES_NUM_FIVE_HHZ]))

        self.amplitudes_pca = []

        data_len = self.phases.shape[0]
        for i in range(self.phases.shape[1]):
            # self.phases[:data_len, i] = dwn_noise(hampel(self.phases[:, i]))[:data_len]
            self.amplitudes[:data_len, i] = dwn_noise(hampel(self.amplitudes[:, i]))[:data_len]

        for i in range(4):
            self.amplitudes_pca.append(
                pca.fit_transform(self.amplitudes[:, i * SUBCARRIES_NUM_FIVE_HHZ:(i + 1) * SUBCARRIES_NUM_FIVE_HHZ]))
        self.amplitudes_pca = np.array(self.amplitudes_pca)
        self.amplitudes_pca = self.amplitudes_pca.reshape((self.amplitudes_pca.shape[1], self.amplitudes_pca.shape[0] * self.amplitudes_pca.shape[2]))

        self.label_keys = list(set(self.labels))
        self.class_to_idx = {
            "standing": 0,
            "walking": 1,
            "get_down": 2,
            "sitting": 3,
            "get_up": 4,
            "lying": 5,
            "no_person": 6
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.window = window_size
        if window_size == -1:
            self.window = self.labels.shape[0] - 1

        self.step = step

    def __getitem__(self, idx):
        if self.window == 0:
            return np.append(self.amplitudes[idx], self.phases[idx]), self.class_to_idx[
                self.labels[idx + self.window - 1]]

        idx = idx * self.step
        all_xs, all_ys = [], []
        # idx = idx * self.window

        for index in range(idx, idx + self.window):
            all_xs.append(np.append(self.amplitudes[index], self.amplitudes_pca[index]))
            # all_ys.append(self.class_to_idx[self.labels[index]])

        return np.array(all_xs), self.class_to_idx[self.labels[idx + self.window - 1]]
        # return np.array(all_xs), np.array(all_ys)

    def __len__(self):
        # return self.labels.shape[0] // self.window
        # return (self.labels.shape[0] - self.window
        return int((self.labels.shape[0] - self.window) // self.step) + 1


if __name__ == '__main__':
    val_dataset = CSIDataset([
        "./dataset/bedroom_lviv/4",
    ])

    dl = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)

    for i in dl:
        print(i[0].shape)
        print(i[1].shape)

        break

    print(val_dataset[0])
