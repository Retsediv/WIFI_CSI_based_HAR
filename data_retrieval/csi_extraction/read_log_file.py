import os

from .read_csi import *


def read_log_file(filename, ignore_endian=0, endian="", check_tones=1):
    f = open(filename, "rb")
    f.seek(0, os.SEEK_END)
    length = f.tell()  # File Length
    print("file length is ", length)

    f.seek(0, 0)  # Default Pos
    if ignore_endian == 0:
        if struct.unpack("B", f.read(1))[0] == 255:
            print("Big-Endian Format")
            endian = ">"
        else:
            print("Little-Endian Format")  # 1 Byte Endian Format
            endian = "<"
    ret = []
    tones = -1
    while (f.tell() < length):
        try:
            csi_inf = unpack_csi_struct(f, endianess=endian)
            if (tones == -1 and check_tones == 1):
                tones = csi_inf.num_tones
            if (check_tones == 1 and csi_inf.num_tones != tones):
                continue  # discarding packet with a different amount of tones
            if (csi_inf.csi == 0):
                print("Ignoring packet, no csi information found")
                continue
            ret.append(csi_inf)
        except:
            print("Corrupt Packet")
            break
    f.close()
    return ret


if __name__ == "__main__":
    file = read_log_file("../../Sandip_Tests/csi.dat")
    print("Done reading")

    for struct in file:
        print(struct.channel)
