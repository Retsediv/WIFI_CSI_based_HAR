/*
 * =====================================================================================
 *       Filename:  sendData.c 
 *
 *    Description:  send packets 
 *        Version:  1.0
 *
 *         Author:  Yaxiong Xie 
 *         Email :  <xieyaxiongfly@gmail.com>
 *   Organization:  WANS group @ Nanyang Technological University 
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 * =====================================================================================
 */
 
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <net/if.h>
#include <netinet/ether.h>
#include <unistd.h>
#include <errno.h>

/* Define the defult destination MAC address */
#define DEFAULT_DEST_MAC0	0x00
#define DEFAULT_DEST_MAC1	0x03
#define DEFAULT_DEST_MAC2	0x7F
#define DEFAULT_DEST_MAC3	0xB0
#define DEFAULT_DEST_MAC4	0x20
#define DEFAULT_DEST_MAC5	0x20
 
#define DEFAULT_IF	        "wlan0"
#define BUF_SIZ	            2048	
 
int main(int argc, char *argv[])
{
	int     sockfd;
    	int     i;
	struct  ifreq if_idx;
	struct  ifreq if_mac;
	int     tx_len = 0,Cnt;
	char    sendbuf[BUF_SIZ];
    	unsigned int DstAddr[6];
	struct  ether_header *eh = (struct ether_header *) sendbuf;
	struct  iphdr *iph = (struct iphdr *) (sendbuf + sizeof(struct ether_header));
	struct  sockaddr_ll socket_address;
	char    ifName[IFNAMSIZ];
	
    if (argc == 1)
    {
        printf("Usage:   %s ifName DstMacAddr NumOfPacketToSend\n",argv[0]);
        printf("Example: %s wlan0 00:7F:5D:3E:4A 100\n",argv[0]);
        exit(0);
    }

	/* Get interface name */
	if (argc > 1)
		strcpy(ifName, argv[1]);
	else
		strcpy(ifName, DEFAULT_IF);

    //dst address seperated by :, example: 00:7F:5D:3E:4A
    if(argc>2)
    {
        sscanf(argv[2],"%x:%x:%x:%x:%x:%x",&DstAddr[0],&DstAddr[1],&DstAddr[2],&DstAddr[3],&DstAddr[4],&DstAddr[5]);
        //printf("DstMacAddr: %02x:%02x:%02x:%02x:%02x:%02x\n",DstAddr[0],DstAddr[1],DstAddr[2],DstAddr[3],DstAddr[4],DstAddr[5]);
    }
    else
    {
        DstAddr[0] = DEFAULT_DEST_MAC0;
        DstAddr[1] = DEFAULT_DEST_MAC1;
        DstAddr[2] = DEFAULT_DEST_MAC2;
        DstAddr[3] = DEFAULT_DEST_MAC3;
        DstAddr[4] = DEFAULT_DEST_MAC4;
        DstAddr[5] = DEFAULT_DEST_MAC5;
    }

    if(argc > 3)
        Cnt = atoi(argv[3]);
    else
        Cnt = 1;
	
 
	/* Open RAW socket to send on */
	if ((sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) == -1) {
	    perror("socket");
	}
 
	/* Get the index of the interface to send on */
	memset(&if_idx, 0, sizeof(struct ifreq));
	strncpy(if_idx.ifr_name, ifName, IFNAMSIZ-1);
	if (ioctl(sockfd, SIOCGIFINDEX, &if_idx) < 0)
	    perror("SIOCGIFINDEX");
	/* Get the MAC address of the interface to send on */
	memset(&if_mac, 0, sizeof(struct ifreq));
	strncpy(if_mac.ifr_name, ifName, IFNAMSIZ-1);
	if (ioctl(sockfd, SIOCGIFHWADDR, &if_mac) < 0)
	    perror("SIOCGIFHWADDR");
 
	/* Construct the Ethernet header */
	memset(sendbuf, 0, BUF_SIZ);
	/* Ethernet header */
	eh->ether_shost[0] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[0];
	eh->ether_shost[1] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[1];
	eh->ether_shost[2] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[2];
	eh->ether_shost[3] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[3];
	eh->ether_shost[4] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[4];
	eh->ether_shost[5] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[5];
	eh->ether_dhost[0] = DstAddr[0];
	eh->ether_dhost[1] = DstAddr[1];
	eh->ether_dhost[2] = DstAddr[2];
	eh->ether_dhost[3] = DstAddr[3];
	eh->ether_dhost[4] = DstAddr[4];
	eh->ether_dhost[5] = DstAddr[5];

    /* Ethertype field */
	eh->ether_type = htons(ETH_P_IP);
	tx_len += sizeof(struct ether_header);
 
	/* Packet data 
     * We just set it to 0xaa you send arbitrary payload you like*/
    for(i=1;i<=1000;i++){
        
	    sendbuf[tx_len++] = 0xaa;
    } 
    printf("Packet Length is: %d,pkt_num is: %d\n",tx_len,Cnt); 
	
    /* Index of the network device */
	socket_address.sll_ifindex = if_idx.ifr_ifindex;
    /* RAW communication*/
    socket_address.sll_family   = PF_PACKET;    
    /* we don't use a protocoll above ethernet layer
     *   ->just use anything here*/
    socket_address.sll_protocol = htons(ETH_P_IP);  
    
    /* ARP hardware identifier is ethernet*/
    socket_address.sll_hatype   = ARPHRD_ETHER;
        
    /* target is another host*/
    socket_address.sll_pkttype  = PACKET_OTHERHOST;
    
    /* address length*/
    socket_address.sll_halen    = ETH_ALEN;
	/* Destination MAC */
	socket_address.sll_addr[0] = DstAddr[0];
	socket_address.sll_addr[1] = DstAddr[1];
	socket_address.sll_addr[2] = DstAddr[2];
	socket_address.sll_addr[3] = DstAddr[3];
	socket_address.sll_addr[4] = DstAddr[4];
	socket_address.sll_addr[5] = DstAddr[5];
 
	/* Send packet */
    for(;Cnt>0;Cnt--)
    {
        /* you set the time interval between two transmitting packets 
         * for example, here we set it to 50 microseconds
         * set to 0 if you don't need it
         */
        if (usleep(50) == -1){
            printf("sleep failed\n");
        }
        if (sendto(sockfd, sendbuf, tx_len, 0, (struct sockaddr*)&socket_address, sizeof(struct sockaddr_ll)) < 0){
            printf("Send failed\n");
        }
    }
	
	return 0;
}
