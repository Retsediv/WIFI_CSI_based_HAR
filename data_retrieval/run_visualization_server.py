import argparse
import csv
import io
import os
from copy import deepcopy

import cv2
import numpy as np
from PyQt6 import QtNetwork, QtWidgets, uic
from pyqtgraph.Qt import QtCore

from csi_extraction import calc_phase_angle, calibrate_amplitude, calibrate_phase, unpack_csi_struct


class UI(QtWidgets.QWidget):
    def __init__(self, app: QtWidgets.QApplication, parent: QtWidgets.QWidget = None, is_5ghz: bool = True):
        super(UI, self).__init__(parent)
        self.app = app
        self.is_5ghz = is_5ghz

        # get and show object and layout
        uic.loadUi('window.ui', self)
        self.setWindowTitle(f"Visualize CSI data {'5GHz' if is_5ghz else '2.4GHz'}")

        self.antenna_pairs, self.carrier, self.amplitude, self.phase = [], [], [], []

        amp = self.box_amp
        amp.setBackground('w')
        amp.setWindowTitle('Amplitude')
        amp.setLabel('bottom', 'Carrier', units='')
        amp.setLabel('left', 'Amplitude', units='')
        amp.setYRange(0, 1, padding=0)  # for normalized amp values, prev range: 0, 90
        amp.setXRange(0, 114 if is_5ghz else 57, padding=0)
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
        phase.setXRange(0, 114 if is_5ghz else 57, padding=0)
        self.penPhase0_0 = phase.plot(pen={'color': (200, 0, 0), 'width': 3})
        self.penPhase0_1 = phase.plot(pen={'color': (200, 200, 0), 'width': 3})
        self.penPhase1_0 = phase.plot(pen={'color': (0, 0, 200), 'width': 3})
        self.penPhase1_1 = phase.plot(pen={'color': (0, 200, 200), 'width': 3})

        self.amp = amp
        self.box_phase = phase

    @QtCore.pyqtSlot()
    def update_plots(self):
        if len(self.amplitude) and len(self.amplitude[0]):
            if len(self.antenna_pairs) > 0:
                self.penAmp0_0.setData(self.carrier, self.amplitude[0])
                self.penPhase0_0.setData(self.carrier, self.phase[0])

            if len(self.antenna_pairs) > 1:
                self.penAmp0_1.setData(self.carrier, self.amplitude[1])
                self.penPhase0_1.setData(self.carrier, self.phase[1])

            if len(self.antenna_pairs) > 2:
                self.penAmp1_0.setData(self.carrier, self.amplitude[2])
                self.penPhase1_0.setData(self.carrier, self.phase[2])

            if len(self.antenna_pairs) > 3:
                self.penAmp1_1.setData(self.carrier, self.amplitude[3])
                self.penPhase1_1.setData(self.carrier, self.phase[3])

        self.process_events()  # force complete redraw for every plot

    def process_events(self):
        self.app.processEvents()


class UDPListener:
    packet_counter = 0

    def __init__(self, save_data_path: str, sock: QtNetwork.QUdpSocket,
                 form: UI, make_photo: bool = False):
        self.save_data_path = save_data_path
        os.makedirs(save_data_path, exist_ok=True)

        self.make_photo = make_photo
        self.cam = cv2.VideoCapture(0) if self.make_photo else None

        self.sock = sock
        sock.readyRead.connect(self.on_datagram_received)

        self.form = form

    def on_datagram_received(self):
        while self.sock.hasPendingDatagrams():
            print("Received new datagram")
            datagram, host, port = self.sock.readDatagram(self.sock.pendingDatagramSize())
            f = io.BytesIO(datagram)
            csi_inf = unpack_csi_struct(f)

            if csi_inf.csi != 0:  # get csi from data packet, save and process for further visualization
                raw_peak_amplitudes, raw_phases, carriers_indexes, antenna_pairs = self.get_csi_raw_data(csi_inf)
                self.calc(raw_peak_amplitudes, raw_phases, carriers_indexes, antenna_pairs)
                self.save_csi_to_file(raw_peak_amplitudes, raw_phases, carriers_indexes)

                if self.make_photo:
                    self.make_photo_and_save()

                self.form.update_plots()
                self.packet_counter += 1

    def get_csi_raw_data(self, csi_inf):
        print("getting csi from the raw data")

        channel = csi_inf.channel
        carriers_num = csi_inf.num_tones
        carriers = []  # just indexes from 0 to "csi_inf.num_tone"

        print("channel: ", channel)
        print("carriers_num: ", carriers_num)

        num_of_antenna_pairs = csi_inf.nc * csi_inf.nr
        antenna_pairs = [(tx_index, rx_index) for tx_index in range(csi_inf.nc) for rx_index in range(csi_inf.nr)]
        raw_phases, raw_peak_amplitudes = [[] for _ in range(num_of_antenna_pairs)], \
                                          [[] for _ in range(num_of_antenna_pairs)]
        print("antenna_pairs: ", antenna_pairs)

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

        return raw_peak_amplitudes, raw_phases, carriers, antenna_pairs

    def calc(self, raw_peak_amplitudes, raw_phases, carriers_indexes, antenna_pairs):  # used for visualization only
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
        self.form.antenna_pairs = antenna_pairs

    def make_photo_and_save(self):
        ret, frame = self.cam.read()

        img_name = "opencv_frame_{}.png".format(UDPListener.packet_counter)
        path_to_image = "{}/{}/{}".format(self.save_data_path, "images", img_name)

        cv2.imwrite(path_to_image, frame)
        print("{} written!".format(path_to_image))

    def save_csi_to_file(self, raw_peak_amplitudes, raw_phases, carriers):
        # print("Saving CSI data to .csv file...")
        filename_csv = "data.csv"
        path_to_csv_file = "{}/{}".format(self.save_data_path, filename_csv)

        with open(path_to_csv_file, 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([*carriers, *raw_peak_amplitudes[0], *raw_peak_amplitudes[1], *raw_peak_amplitudes[2],
                             *raw_peak_amplitudes[3], *raw_phases[0], *raw_phases[1], *raw_phases[2], *raw_phases[3]])


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualization server that listens to any incoming packets, plot them and store.\n"
                    "Only supports up to 4 antenna pairs at the moment.",
        prog="python run_visualization_server.py"
    )

    parser.add_argument("-f", "--frequency", help="Frequency on which both routes are operating",
                        choices=['2400MHZ', '5000MHZ'], default='5000MHZ')
    parser.add_argument("-s", "--save_path", help="path to the folder where to save the incoming data",
                        default="./tmp", type=str)
    parser.add_argument("-p", "--port", help="port to listen to", default=1234, type=int)
    parser.add_argument("--photo", help="make webcam photo for each data packet?", default=False, type=bool)

    return parser


def run_app() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    app = QtWidgets.QApplication([])

    try:
        udp_socket = QtNetwork.QUdpSocket()
        udp_socket.bind(QtNetwork.QHostAddress.SpecialAddress.Any, args.port)

        form = UI(app=app, is_5ghz=(args.frequency == '5000MHZ'))
        form.show()

        listener = UDPListener(save_data_path=args.save_path, sock=udp_socket,
                               form=form, make_photo=args.photo)
        app.exec()

    except KeyboardInterrupt as e:
        try:
            listener.cam.release()
        except Exception as e_cam:
            pass

        print("KeyboardInterrupt. Finishing the program...")


if __name__ == '__main__':
    run_app()
