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

import numpy as np
import PyQt4.uic as uic  # need this to load ui file from QT Creator
from PyQt4 import QtNetwork
from pyqtgraph.Qt import QtCore, QtGui

from csi_extraction import unpack_csi_struct
import csv

app = QtGui.QApplication([])


class UDP_listener():

    def __init__(self, carrier, amplitudes, sock, form):
        self.carrier = carrier
        self.amplitude = amplitudes
        self.sock = sock
        sock.readyRead.connect(self.datagramReceived)
        self.form = form

    def datagramReceived(self):
        while self.sock.hasPendingDatagrams():
            datagram, host, port = self.sock.readDatagram(self.sock.pendingDatagramSize())
            f = io.BytesIO(datagram)
            csi_inf = unpack_csi_struct(f)
            if (csi_inf.csi != 0):
                self.calc(csi_inf)

    # def calc(self, csi_inf):
    #     channel = csi_inf.channel
    #     carriers = csi_inf.num_tones
    #     carrier = []
    #     phase = []
    #     amplitude = []
    #
    #     phase0_0 = []
    #     phase0_1 = []
    #     phase1_0 = []
    #     phase1_1 = []
    #
    #     for i in range(0, carriers):
    #         if i == 0:
    #             print("=== CSI FULL DATA ===")
    #             print(csi_inf.csi[i])
    #             print("\n\n")
    #
    #         p = csi_inf.csi[i][0][0]
    #         imag = p.imag
    #         real = p.real
    #         peak_amplitude = np.sqrt(np.power(real, 2) + np.power(imag, 2))
    #         carrier.append(i)
    #         amplitude.append(peak_amplitude)
    #         phase.append(np.angle(p))
    #
    #     self.form.carrier = carrier
    #     self.form.amplitude = ((np.array(amplitude) - np.min(amplitude)) / (
    #             np.max(amplitude) - np.min(amplitude)) * csi_inf.rssi)  # normalized phase
    #     self.form.phase = np.unwrap(phase)

    def calc(self, csi_inf):
        channel = csi_inf.channel
        carriers = csi_inf.num_tones
        carrier = []  # just indexes from 0 to "csi_inf.num_tone"

        phase = [[], [], [], []]
        amplitude = [[], [], [], []]

        for i in range(0, carriers):
            for enum_index, (tr_i, rc_i) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                p = csi_inf.csi[i][tr_i][rc_i]
                imag = p.imag
                real = p.real
                peak_amplitude = np.sqrt(np.power(real, 2) + np.power(imag, 2))
                phase[enum_index].append(np.angle(p))
                amplitude[enum_index].append(peak_amplitude)

            carrier.append(i)

        self.form.carrier = carrier
        for i in range(4):
            amplitude[i] = ((np.array(amplitude[i]) - np.min(amplitude[i])) / (
                    np.max(amplitude[i]) - np.min(amplitude[i])) * csi_inf.rssi)  # normalized phase
        self.form.amplitude = amplitude

        # Calibrate phases
        final_phases = []

        for p_i in range(4):
            current_phase = phase[p_i]
            subcarries = 56
            calibrated_phase = [0] * len(current_phase)
            m = range((-1) * subcarries + 2, subcarries - 2)
            tp = [0] * len(current_phase)
            diff = 0
            mu = np.pi
            tp[0] = current_phase[0]

            for i in range(1, len(current_phase)):
                if current_phase[i] - current_phase[i - 1] > mu:
                    diff += 1

                tp[i] = current_phase[i] - diff * 2 * np.pi

            k = (tp[-1] - tp[0]) / (m[len(current_phase) - 1] - m[0])
            b = sum(tp) / len(current_phase)

            for i in range(len(current_phase)):
                calibrated_phase[i] = tp[i] - k * m[i] - b

            final_phases.append(calibrated_phase)

        self.form.phase = final_phases
        # self.form.phase = np.unwrap(phase)


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
        amp.setYRange(0, 90, padding=0)
        amp.setXRange(0, 56, padding=0)
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
        phase.setXRange(0, 56, padding=0)
        self.penPhase0_0 = phase.plot(pen={'color': (200, 0, 0), 'width': 3})
        self.penPhase0_1 = phase.plot(pen={'color': (200, 200, 0), 'width': 3})
        self.penPhase1_0 = phase.plot(pen={'color': (0, 0, 200), 'width': 3})
        self.penPhase1_1 = phase.plot(pen={'color': (0, 200, 200), 'width': 3})

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(20)
        self.amp = amp
        self.box_phase = phase

    def save_to_file(self, carrier, phase, amplitude):
        # print("save to file")

        # TODO: photo shooting
        filename_csv = "./ruh.csv"
        with open(filename_csv, 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([*phase[0], *phase[1], *phase[2], *phase[3], *amplitude[0], *amplitude[1], *amplitude[2],
                             *amplitude[3]])

    @QtCore.pyqtSlot()
    def update_plots(self):
        # print("UPDATE slots!")
        # print(self.amplitude)

        self.penAmp0_0.setData(self.carrier, self.amplitude[0])
        self.penAmp0_1.setData(self.carrier, self.amplitude[1])
        self.penAmp1_0.setData(self.carrier, self.amplitude[2])
        self.penAmp1_1.setData(self.carrier, self.amplitude[3])

        self.penPhase0_0.setData(self.carrier, self.phase[0])
        self.penPhase0_1.setData(self.carrier, self.phase[1])
        self.penPhase1_0.setData(self.carrier, self.phase[2])
        self.penPhase1_1.setData(self.carrier, self.phase[3])

        self.save_to_file(self.carrier, self.phase, self.amplitude)

        self.process_events()  # force complete redraw for every plot

    def process_events(self):
        self.app.processEvents()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    udp_port = 1234
    udpSocket = QtNetwork.QUdpSocket()
    udpSocket.bind(udp_port, QtNetwork.QUdpSocket.ShareAddress)

    form = UI(app=app)
    form.show()

    e = UDP_listener([], [], sock=udpSocket, form=form)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
