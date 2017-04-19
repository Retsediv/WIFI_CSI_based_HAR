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
import sys
import os
from socket import *
import struct
from struct import unpack, pack
from time import clock, time, sleep


import argparse
import collections
import numpy as np
import copy

from decimal import Decimal
from collections import deque

from pyqtgraph.Qt import QtGui, QtCore
from PyQt4 import QtNetwork
import pyqtgraph as pg
# need this to load ui file from QT Creator
import PyQt4.uic as uic

app = QtGui.QApplication([])

import random

from threading import Thread

from read_csi import unpack_csi_struct
import io

class UDP_listener():

    def __init__(self, carrier, amplitudes, sock, form):
        self.carrier      =   carrier
        self.amplitude    =   amplitudes
        self.sock = sock
        sock.readyRead.connect(self.datagramReceived)
        self.form = form

    def datagramReceived(self):
        while self.sock.hasPendingDatagrams():
            datagram, host, port = self.sock.readDatagram(self.sock.pendingDatagramSize())
            f = io.BytesIO(datagram)
            csi_inf = unpack_csi_struct(f)
            if(csi_inf.csi != 0):
                self.calc(csi_inf)

    def calc(self, csi_inf):
        channel = csi_inf.channel
        carriers = csi_inf.num_tones
        carrier = []
        phase = []
        amplitude = []
        for i in range(0, carriers):
            p = csi_inf.csi[i][0][0]
            imag = p.imag
            real = p.real
            peak_amplitude = np.sqrt(np.power(real, 2)+ np.power(imag,2))
            carrier.append(i)
            amplitude.append(peak_amplitude)
            phase.append(np.angle(p))

        self.form.carrier = carrier
        self.form.amplitude = ((np.array(amplitude) - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude)) * csi_inf.rssi) #normalized phase
        self.form.phase = np.unwrap(phase)

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
        amp.setXRange(0, 114, padding=0)
        self.penAmp = amp.plot(pen={'color': (0, 100, 0), 'width': 3})

        phase = self.box_phase
        phase.setBackground('w')
        phase.setWindowTitle('Phase')
        phase.setLabel('bottom', 'Carrier', units='')
        phase.setLabel('left', 'Phase', units='')
        phase.setYRange(-np.pi, np.pi, padding=0)
        phase.setXRange(0, 114, padding=0)
        self.penPhase = phase.plot(pen={'color': (0, 100, 0), 'width': 3})

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(20)
        self.amp = amp
        self.box_phase = phase

    @QtCore.pyqtSlot()
    def update_plots(self):
        self.penAmp.setData(self.carrier, self.amplitude)
        self.penPhase.setData(self.carrier, self.phase)
        self.process_events()  ## force complete redraw for every plot

    def process_events(self):
        self.app.processEvents()

## Start Qt event loop unless running in interactive mode.
if (__name__ == '__main__'):
    import sys

    udp_port=1234
    udpSocket = QtNetwork.QUdpSocket()
    udpSocket.bind( udp_port, QtNetwork.QUdpSocket.ShareAddress)

    form = UI(app=app)
    form.show()
    
    e=UDP_listener([], [], sock=udpSocket, form=form)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



