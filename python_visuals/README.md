# ath9k_csi_userspace
Additions to the atheros csi user space tools

Visualize CSI requires pyqt.
![Screenshot](screenshot.png)

*Sample Setup for visualize CSI*
The setup consists of three devices. Two SoC's and one PC/Notebook

1. The first SoC (A) acts as an access point (AP). The other SoC (B) and the PC connect to that AP.
2. A starts the recv_csi tool from [HERE](https://github.com/NovelSense/Atheros-CSI-Tool-UserSpace-APP/tree/master/recvCSI) with the IP Address of the PC and the port 1234 `./recv_csi <PC IP> 1234`
3. The PC starts the visualie_csi.py script.
4. B starts the send_data tool from [HERE](https://github.com/NovelSense/Atheros-CSI-Tool-UserSpace-APP/tree/master/sendData) and starts sending packages. Those packages should now be received by the PC.

*Troubleshooting*:
If the packages are not received by the PC there might be issues with the firewall.
    