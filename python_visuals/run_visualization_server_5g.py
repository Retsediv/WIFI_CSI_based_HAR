#!/usr/bin/python -i
# @file visualize_csi.py
#
# @brief Simple visualization based on udp packet data
#
# uses: pyqtgraph, pyqt
#
# Copyright 2016 NovelSense UG
# Authors: Niklas Saenger, Markus Scholz
#
# NOTICE: All information contained herein is, and remains the
# property of NovelSense UG and its suppliers, if any.  The
# intellectual and technical concepts contained herein are
# proprietary to NovelSense UG and its suppliers, and are protected
# by trade secret or copyright law. Dissemination of this
# information or reproduction of this material is strictly forbidden
# unless prior written permission is obtained from NovelSense UG.
# ------------------------------------------------------------------------------------------

import io

import cv2
import numpy as np
import PyQt4.uic as uic  # need this to load ui file from QT Creator
from PyQt4 import QtNetwork
from pyqtgraph.Qt import QtCore, QtGui

from csi_extraction import unpack_csi_struct, calc_phase_angle
from csi_extraction import calibrate_amplitude, calibrate_phase
from copy import deepcopy
import csv

app = QtGui.QApplication([])

# cam = cv2.VideoCapture(0)
PATH_TO_DATA_FOLDER = "./experiments/tmp"
# PATH_TO_DATA_FOLDER = "./experiments/tmp"


class UDP_listener():
    packet_counter = 0

    def __init__(self, carrier, amplitudes, sock, form):
        self.carrier = carrier
        self.amplitude = amplitudes
        self.sock = sock
        sock.readyRead.connect(self.datagramReceived)
        self.form = form

    def datagramReceived(self):
        while self.sock.hasPendingDatagrams():
            print("received datagram")
            datagram, host, port = self.sock.readDatagram(self.sock.pendingDatagramSize())
            f = io.BytesIO(datagram)
            csi_inf = unpack_csi_struct(f)

            if csi_inf.csi != 0:  # get csi from data packet, save and process for further visualization
                raw_peak_amplitudes, raw_phases, carriers_indexes = self.get_csi_raw_data(csi_inf)
                self.calc(raw_peak_amplitudes, raw_phases, carriers_indexes)
                self.save_csi_to_file(raw_peak_amplitudes, raw_phases, carriers_indexes)
                # self.make_photo_and_save()

                self.form.update_plots()

    def get_csi_raw_data(self, csi_inf):
        print("def get csi raw data")

        channel = csi_inf.channel
        carriers_num = csi_inf.num_tones
        carriers = []  # just indexes from 0 to "csi_inf.num_tone"

        print("channel: ", channel)
        print("carriers_num: ", carriers_num)

        # antenna_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]  # for 2 antennas
        antenna_pairs = [(0, 0)]  # for 1 antenna
        raw_phases, raw_peak_amplitudes = [[] * len(antenna_pairs)], [[] * len(antenna_pairs)]

        for i in range(0, carriers_num):
            for enum_index, (tr_i, rc_i) in enumerate(antenna_pairs):
                # print("csi_len ", csi_inf.csi_len)
                # print("csi_inf.csi[i]: ", csi_inf)
                p = csi_inf.csi[i][tr_i][rc_i]
                imag, real = p.imag, p.real

                peak_amplitude = np.sqrt(np.power(real, 2) + np.power(imag, 2))  # calculate peak amplitude
                phase_angle = calc_phase_angle(p)  # calculate phase angle

                raw_peak_amplitudes[enum_index].append(peak_amplitude)
                raw_phases[enum_index].append(phase_angle)
            carriers.append(i)

        return raw_peak_amplitudes, raw_phases, carriers

    def calc(self, raw_peak_amplitudes, raw_phases, carriers_indexes):  # used for visualization only
        amplitude, phase = deepcopy(raw_peak_amplitudes), deepcopy(raw_phases)

        # Update form carriers indexes
        self.form.carrier = carriers_indexes

        # Calibrate amplitude and update form amplitude values
        for i in range(len(amplitude)):
            amplitude[i] = calibrate_amplitude(amplitude[i], 1).tolist()
        self.form.amplitude = amplitude

        # Calibrate phase and update form phase values
        for i in range(len(phase)):
            phase[i] = calibrate_phase(phase[i]).tolist()
        self.form.phase = phase

    @staticmethod
    def make_photo_and_save():
        # ret, frame = cam.read()
        #
        img_name = "opencv_frame_{}.png".format(UDP_listener.packet_counter)
        # path_to_image = "{}/{}/{}".format(PATH_TO_DATA_FOLDER, "images", img_name)
        # cv2.imwrite(path_to_image, frame)
        #
        # UDP_listener.packet_counter += 1
        # print("{} written!".format(path_to_image))

    @staticmethod
    def save_csi_to_file(raw_peak_amplitudes, raw_phases, carriers):
        # print("Saving CSI data to .csv file...")

        filename_csv = "data.csv"
        path_to_csv_file = "{}/{}".format(PATH_TO_DATA_FOLDER, filename_csv)

        with open(path_to_csv_file, 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # writer.writerow([*carriers, *raw_peak_amplitudes[0], *raw_peak_amplitudes[1], *raw_peak_amplitudes[2],
            #                  *raw_peak_amplitudes[3], *raw_phases[0], *raw_phases[1], *raw_phases[2], *raw_phases[3]])
            writer.writerow([*carriers, *raw_peak_amplitudes[0], *raw_phases[0]])


class UI(QtGui.QWidget):
    def __init__(self, app, parent=None):
        super(UI, self).__init__(parent)
        self.app = app

        # get and show object and layout
        uic.loadUi('window.ui', self)

        self.setWindowTitle("Visualize CSI")

        self.carrier = []
        self.amplitude = []
        self.phase = []

        amp = self.box_amp
        amp.setBackground('w')
        amp.setWindowTitle('Amplitude')
        amp.setLabel('bottom', 'Carrier', units='')
        amp.setLabel('left', 'Amplitude', units='')
        amp.setYRange(0, 1, padding=0)  # for normalized amp values, prev range: 0, 90
        amp.setXRange(0, 114, padding=0)
        self.penAmp0_0 = amp.plot(pen={'color': (200, 0, 0), 'width': 3})
        self.penAmp0_1 = amp.plot(pen={'color': (200, 200, 0), 'width': 3})
        self.penAmp1_0 = amp.plot(pen={'color': (0, 0, 200), 'width': 3})
        self.penAmp1_1 = amp.plot(pen={'color': (0, 200, 200), 'width': 3})

        phase = self.box_phase
        phase.setBackground('w')
        phase.setWindowTitle('Phase')
        phase.setLabel('bottom', 'Carrier', units='')
        phase.setLabel('left', 'Phase', units='')
        phase.setYRange(-np.pi, np.pi, padding=0)
        phase.setXRange(0, 114, padding=0)
        self.penPhase0_0 = phase.plot(pen={'color': (200, 0, 0), 'width': 3})
        self.penPhase0_1 = phase.plot(pen={'color': (200, 200, 0), 'width': 3})
        self.penPhase1_0 = phase.plot(pen={'color': (0, 0, 200), 'width': 3})
        self.penPhase1_1 = phase.plot(pen={'color': (0, 200, 200), 'width': 3})

        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.update_plots)
        # self.timer.start(20)

        self.amp = amp
        self.box_phase = phase

    @QtCore.pyqtSlot()
    def update_plots(self):
        if len(self.amplitude) and len(self.amplitude[0]):
            self.penAmp0_0.setData(self.carrier, self.amplitude[0])
            # self.penAmp0_1.setData(self.carrier, self.amplitude[1])
            # self.penAmp1_0.setData(self.carrier, self.amplitude[2])
            # self.penAmp1_1.setData(self.carrier, self.amplitude[3])

            self.penPhase0_0.setData(self.carrier, self.phase[0])
            # self.penPhase0_1.setData(self.carrier, self.phase[1])
            # self.penPhase1_0.setData(self.carrier, self.phase[2])
            # self.penPhase1_1.setData(self.carrier, self.phase[3])

        self.process_events()  # force complete redraw for every plot

    def process_events(self):
        self.app.processEvents()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    try:
        udp_port = 1234
        udpSocket = QtNetwork.QUdpSocket()
        udpSocket.bind(udp_port, QtNetwork.QUdpSocket.ShareAddress)

        form = UI(app=app)
        form.show()

        e = UDP_listener([], [], sock=udpSocket, form=form)

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    except KeyboardInterrupt as e:
        # cam.release()
        print("Finish!")
