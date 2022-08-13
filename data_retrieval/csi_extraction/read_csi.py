import io
import struct
from math import atan2

BITS_PER_BYTE = 8
BITS_PER_SYMBOL = 10
bitmask = (1 << BITS_PER_SYMBOL) - 1
DEBUG = 1


class csi_struct:
    pass


def unpack_csi_struct(f, endianess='>'):  # Big-Endian as Default Value
    csi_inf = csi_struct()
    csi_inf.field_len = struct.unpack(endianess + 'H', f.read(2))[0]  # Block Length     1Byte
    csi_inf.timestamp = struct.unpack(endianess + 'Q', f.read(8))[0]  # TimeStamp      8Byte
    csi_inf.csi_len = struct.unpack(endianess + 'H', f.read(2))[0]  # csi_len        2Byte
    csi_inf.channel = struct.unpack(endianess + 'H', f.read(2))[0]  # tx             2Byte
    csi_inf.err_info = struct.unpack(endianess + 'B', f.read(1))[0]  # err_info       1Byte
    csi_inf.rate = struct.unpack(endianess + 'B', f.read(1))[0]  # noisefloor     1Byte
    csi_inf.noise_floor = struct.unpack(endianess + 'B', f.read(1))[0]  # rate           1Byte
    csi_inf.bw = struct.unpack(endianess + 'B', f.read(1))[0]  # bandWidth      1Byte
    csi_inf.num_tones = struct.unpack(endianess + 'B', f.read(1))[0]  # num_tones      1Byte
    csi_inf.nr = struct.unpack(endianess + 'B', f.read(1))[0]  # nr             1Byte
    csi_inf.nc = struct.unpack(endianess + 'B', f.read(1))[0]  # nc             1Byte
    csi_inf.rssi = struct.unpack(endianess + 'B', f.read(1))[0]  # rssi           1Byte
    csi_inf.rssi1 = struct.unpack(endianess + 'B', f.read(1))[0]  # rssi1          1Byte
    csi_inf.rssi2 = struct.unpack(endianess + 'B', f.read(1))[0]  # rssi2          1Byte
    csi_inf.rssi3 = struct.unpack(endianess + 'B', f.read(1))[0]  # rssi3          1Byte
    csi_inf.payload_len = struct.unpack(endianess + 'H', f.read(2))[
        0]  # payload_len    2Byte Total: 27Byte + csi_len + payload_len

    if csi_inf.csi_len > 0:
        csi_buf = f.read(csi_inf.csi_len)  # csi        csi_len
        csi_inf.csi = read_csi(csi_buf, csi_inf.num_tones, csi_inf.nc, csi_inf.nr, csi_inf.csi_len, endianess)
    else:
        csi_inf.csi = 0

    if csi_inf.payload_len > 0:
        payload_buf = f.read(csi_inf.payload_len)  # payload_len    payload_len
    else:
        payload_buf = 0

    return csi_inf


def read_csi(csi_buf, num_tones, nc, nr, csi_len, endianess):
    csi = []
    buf = io.BytesIO(csi_buf)

    bits_left = 16
    cur_data = struct.unpack(endianess + 'B', buf.read(1))[0]  # Read 16Bits at a Time
    cur_data += (struct.unpack(endianess + 'B', buf.read(1))[0] << BITS_PER_BYTE)

    print("read csi")
    print("num_tones: ", num_tones)
    print("nc: ", nc)
    print("nr: ", nr)
    print()

    for i in range(0, num_tones):
        tones = []
        for nc_idx in range(0, nc):
            A = []
            for nr_idx in range(0, nr):
                if (bits_left - BITS_PER_SYMBOL) < 0:
                    new_bits = struct.unpack(endianess + 'B', buf.read(1))[0]
                    new_bits += (struct.unpack(endianess + 'B', buf.read(1))[0] << BITS_PER_BYTE)
                    # print(new_bits)
                    cur_data += new_bits << bits_left
                    bits_left += 16

                _imag = cur_data & bitmask
                imag = signbit_convert(_imag, BITS_PER_SYMBOL)

                bits_left -= BITS_PER_SYMBOL
                cur_data = cur_data >> BITS_PER_SYMBOL

                if (bits_left - BITS_PER_SYMBOL) < 0:
                    new_bits = struct.unpack(endianess + 'B', buf.read(1))[0]
                    new_bits += (struct.unpack(endianess + 'B', buf.read(1))[0] << BITS_PER_BYTE)
                    # print(new_bits)
                    cur_data += new_bits << bits_left
                    bits_left += 16

                _real = cur_data & bitmask
                real = signbit_convert(_real, BITS_PER_SYMBOL)

                bits_left -= BITS_PER_SYMBOL
                cur_data = cur_data >> BITS_PER_SYMBOL

                # print("R:", real, "I:", imag)
                A.append(complex(real, imag))
            tones.append(A)
        csi.append(tones)

        # print("csi: ", csi)
    return csi


def signbit_convert(data, maxbit):
    if data & (1 << (maxbit - 1)):
        data -= (1 << maxbit)
    return data


def calc_frequency(basefrequency, c, num_tones):
    per_side = num_tones / 2
    step = 0.3125
    if (c < per_side):  # Carrier is on left side of centerfrequency
        freq = basefrequency - (per_side - c) * step
    else:
        freq = basefrequency + (c + 1 - per_side) * step
    return freq


def calc_phase_angle(iq, unwrap=0):
    imag = iq.imag
    real = iq.real
    if unwrap == 1:
        return atan2(imag, real)
    return atan2(imag, real)
