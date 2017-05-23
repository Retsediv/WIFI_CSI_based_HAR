This repository is a fork from the Atheros-CSI-Tool. The following changes were made to the original:
  * Added [python](https://github.com/NovelSense/Atheros-CSI-Tool-UserSpace-APP/tree/master/python) folder: It contains scripts to plot and live visualize the csi data.
  * [recvCSI](https://github.com/NovelSense/Atheros-CSI-Tool-UserSpace-APP/tree/master/recvCSI): Modified the recvCSI tool to send the received CSI packages to the live visualization script.
  * [sendData](https://github.com/NovelSense/Atheros-CSI-Tool-UserSpace-APP/tree/master/sendData): Modified the sendCSI tool to send in a continious time intervall

---
The user-space applications for our Atheros-CSI-TOOL

Please visit our maintainance page http://pdcc.ntu.edu.sg/wands/Atheros/ for detailed infomration.

If you want more details on how to use this tool, please request the documentation http://pdcc.ntu.edu.sg/wands/Atheros/install/install_info.html.

Change Log, we now support one transmitter multiple receivers at the same time. One packet transmitted can be simultaneously received by multiple receivers and the CSI will be calculated accordingly. More detail can be found from our maintainance page. http://pdcc.ntu.edu.sg/wands/Atheros/
