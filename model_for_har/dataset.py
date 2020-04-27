import os

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from data_calibration import calibrate_phase, calibrate_amplitude

DATASET_FOLDER = "../dataset"
DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

SUBCARRIES_NUM_TWO_HHZ = 56
SUBCARRIES_NUM_FIVE_HHZ = 114


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

    final_amplitudes, final_phases, final_labels = np.empty((0, antenna_pairs*SUBCARRIES_NUM_FIVE_HHZ)), \
                                                   np.empty((0, antenna_pairs*SUBCARRIES_NUM_FIVE_HHZ)), \
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

    def __init__(self, csv_files):
        self.amplitudes, self.phases, self.labels = read_all_data_from_files(csv_files)

        self.phases = calibrate_phase(self.phases)
        self.amplitudes = calibrate_amplitude(self.amplitudes)

        self.label_keys = list(set(self.labels))
        self.class_to_idx = {
            self.label_keys[0]: 0,
            self.label_keys[1]: 1,
            self.label_keys[2]: 2,
            self.label_keys[3]: 3,
            self.label_keys[4]: 4,
            self.label_keys[5]: 5,
            self.label_keys[6]: 6
        }

        self.window = 16

    def __getitem__(self, idx):
        all_xs = []

        for i in range(idx, idx + self.window):
            all_xs.append(np.append(self.amplitudes[idx], self.phases[idx]))

        y = self.class_to_idx[self.labels[idx + self.window - 1]]

        return np.array(all_xs), y

    def __len__(self):
        return self.labels.shape[0] - self.window


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
