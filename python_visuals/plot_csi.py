import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from csi_extraction import read_log_file, read_csi, calc_frequency, calc_phase_angle


def plot_phases(csi_structs, unwrap=0):
    if len(csi_structs) == 0:
        return

    all_angles = []
    for csi_struct in csi_structs:
        print("csi_struct.channel: ", csi_struct.channel)
        plt_y = []
        plt_x = []
        print("csi_struct.num_tones: ", csi_struct.num_tones)
        for i in range(0, csi_struct.num_tones):
            angle = calc_phase_angle(csi_struct.csi[i][0][0])  # IMPORTANT: here they calculate phase only between
                                                               #            0st and 0st antenna

            freq = calc_frequency(csi_struct.channel, i, csi_struct.num_tones)
            plt_x.append(freq)
            plt_y.append(angle)

        plt.plot(plt_x, np.unwrap(plt_y))
        all_angles.append(plt_y)

    all_angles = np.unwrap(all_angles)
    y = np.mean(all_angles, axis=0)
    plt.plot(plt_x, y, '-o')
    plt.ylim([-6, 6])
    pdf.savefig()
    plt.close()


if __name__ == "__main__":
    dirpath = ""
    csi_data_filenames = [
        "./data/sample_bigEndian.dat"
    ]

    with PdfPages('plot_csi.pdf') as pdf:
        for f in csi_data_filenames:
            print("Plotting ", f)
            structs = read_log_file(dirpath + f)[:10]  # Plot only 10 packets

            # for key in structs[0]:
            #     print(key, " : ", structs[0][key])

            print(len(structs), " Structs found")
            plot_phases(structs, unwrap=1)
