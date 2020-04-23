#!/usr/bin/python -i
# @file plot_phase.py
#
# @brief Reading logfiles created by userspace tool
#
#
# Copyright 2017 NovelSense UG
# Authors: Niklas Saenger
#
# NOTICE: All information contained herein is, and remains the
# property of NovelSense UG and its suppliers, if any.  The
# intellectual and technical concepts contained herein are
# proprietary to NovelSense UG and its suppliers, and are protected
# by trade secret or copyright law. Dissemination of this
# information or reproduction of this material is strictly forbidden
# unless prior written permission is obtained from NovelSense UG.
# ------------------------------------------------------------------------------------------

from .csi_extraction import read_log_file, read_csi, calc_frequency, calc_phase_angle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_phase(csi_struct, unwrap=0):
    plt_y = []
    plt_x = []
    for i in range(0, csi_struct.num_tones):
        angle = calc_phase_angle(csi_struct.csi[i][0][0], unwrap)
        freq = calc_frequency(csi_struct.channel, i, csi_struct.num_tones)
        plt_y.append(freq)
        plt_x.append(angle)
    if unwrap == 1:
        plt_x = np.unwrap(plt_x)
    plt.plot(plt_y, plt_x, '-o')
    plt.ylim([-np.pi, np.pi])
    pdf.savefig()
    plt.close()


def plot_phases(csi_structs, unwrap=0):
    if len(csi_structs) == 0:
        return

    x = []
    for csi_struct in csi_structs:
        print(csi_struct.channel)
        plt_y = []
        plt_x = []
        for i in range(0, csi_struct.num_tones):
            angle = calc_phase_angle(csi_struct.csi[i][0][0])
            freq = calc_frequency(csi_struct.channel, i, csi_struct.num_tones)
            plt_x.append(freq)
            plt_y.append(angle)
        plt.plot(plt_x, np.unwrap(plt_y))
        x.append(plt_y)

    x = np.unwrap(x)
    x = np.mean(x, axis=0)
    plt.plot(plt_x, x, '-o')
    plt.ylim([-6, 6])
    pdf.savefig()
    plt.close()


if __name__ == "__main__":
    dirpath = ""
    filename = ["0_0_1.dat"]
    with PdfPages('out.pdf') as pdf:
        for f in filename:
            print("Plotting ", f)
            structs = read_log_file(dirpath + f)[:10]  # Plot only 10 packets
            print(len(structs), " Structs found")
            plot_phases(structs, unwrap=1)
