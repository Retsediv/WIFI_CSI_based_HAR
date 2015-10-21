#!/bin/bash
#sudo restart networking
sudo nmcli nm wifi off
sudo rfkill unblock wlan

sudo ifconfig wlan0 down
sudo ifconfig wlan0 10.10.0.1/24 up
sleep 1

sudo service isc-dhcp-server restart
sudo ./hostapd ./hostapd.conf
