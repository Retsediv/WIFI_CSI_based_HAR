import os
import socket
import struct
import argparse


def run_client(filename: str, host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if not os.path.exists(filename):
        print(f"There is no such file: {filename}")
        return

    with open(filename, "rb") as f:
        f.seek(0, os.SEEK_END)
        length = f.tell()
        print(f"File length: {length}")

        f.seek(0, 0)

        if struct.unpack("B", f.read(1))[0] == 255:
            print("Data is in Big-Endian Format")
            endian = ">"
        else:
            print("Data is in Little-Endian Format")  # 1 Byte Endian Format
            endian = "<"

        while f.tell() < length:
            prev = f.tell()
            block_length = struct.unpack(endian + "H", f.read(2))[0]
            block_length += 2
            f.seek(prev, 0)
            data = f.read(block_length)
            sock.sendto(data, (host, port))
            # time.sleep(0.1)
            input("Press Enter to continue...")

            # if(f.tell() == length):  # Go to the beginning of the file and start again
            #    f.seek(0,0)
            #    f.read(1)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dummy client that is sending CSI data packet to the server\n"
                    "Try data/sample_csi_packet_big_endian.day",
        prog="python run_test_client.py"
    )

    parser.add_argument("filename", help="path to the file with a sample data", type=str)
    parser.add_argument("--host", help="host ip to send data", default="127.0.0.1", type=str)
    parser.add_argument("--port", help="host port", default=1234, type=int)

    return parser


if __name__ == "__main__":
    parser = init_argparse()  # Initialize args parser
    args = parser.parse_args()  # Read arguments from command line

    run_client(args.filename, args.host, args.port)
