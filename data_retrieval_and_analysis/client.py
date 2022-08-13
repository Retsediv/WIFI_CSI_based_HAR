#!/usr/bin/python -i
# @file client.py
#
# @brief Simple visualization based on udp packet data
#
# uses: socket, strcut
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
import os
import socket  # Import socket module
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

filename = ".."
try:
    f = open(filename, "rb")
except IOError as e:
    print("I/O Error", e)
f.seek(0, os.SEEK_END)
length = f.tell()  # File Length
print("file length is ", length)
f.seek(0, 0)  # Default Pos

if struct.unpack("B", f.read(1))[0] == 255:
    print("Big-Endian Format")
    endian = ">"
else:
    print("Little-Endian Format")  # 1 Byte Endian Format
    endian = "<"

while f.tell() < length:
    prev = f.tell()
    block_length = struct.unpack(endian + "H", f.read(2))[0]
    block_length += 2
    f.seek(prev, 0)
    data = f.read(block_length)
    sock.sendto(data, ("127.0.0.1", 1234))
    # time.sleep(0.1)
    input("Press Enter to continue...")
    # if(f.tell() == length):
    #    f.seek(0,0)
    #    f.read(1)
