#!/bin/bash
sudo /bin/bash -c "
restart isc-dhcp-server
echo 1 > /proc/sys/net/ipv4/ip_forward
iptables -t nat -A POSTROUTING -s 10.10.0.0/16 -o ppp0 -j MASQUERADE
./hostapd  ./hostapd.conf -B
"
