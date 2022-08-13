/*
 * =====================================================================================
 *       Filename:  main.c
 *
 *    Description:  Here is an example for receiving CSI matrix
 *                  Basic CSi procesing fucntion is also implemented and called
 *                  Check csi_fun.c for detail of the processing function
 *        Version:  1.0
 *
 *         Author:  Yaxiong Xie
 *         Email :  <xieyaxiongfly@gmail.com>
 *   Organization:  WANDS group @ Nanyang Technological University
 *
 *   Copyright (c)  WANDS group @ Nanyang Technological University
 * =====================================================================================
 *
 * =====================================================================================
 *       Version:  1.1
 *        Author:  Andrii Zhuravchak
 *        Email :  <zhuravchak@ucu.edu.ua>
 *  Organization:  Ukrainian Catholic University
 * =====================================================================================
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>

// UDP Stuff begins Here
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>

#include "csi_fun.h"

#define BUFSIZE 4096

int quit;
unsigned char buf_addr[BUFSIZE];
unsigned char data_buf[1500];

COMPLEX csi_matrix[3][3][114];
csi_struct* csi_status;

void sig_handler(int signo) {
    if (signo == SIGINT)
        quit = 1;
}

int main(int argc, char* argv[]) {
    FILE* fp;
    int fd;
    int i;
    int total_msg_cnt, cnt;
    int log_flag;
    unsigned char endian_flag;
    u_int16_t buf_len;

    /*Network Stuff*/
    int clientSocket, portNum, nBytes, data_send;
    struct sockaddr_in serverAddr;
    socklen_t addr_size;

    log_flag = 0;
    csi_status = (csi_struct*) malloc(sizeof(csi_struct));
    /* check usage */
    if (1 == argc) {
        /* If you want to log the CSI for off-line processing,
         * you need to specify the name of the output file
         */
        log_flag = 0;
        printf("/**************************************/\n");
        printf("/*   Usage: recv_csi_nolog_better_output  <udp server(IP)> <udp port> <output_file>    */\n");
        printf("/**************************************/\n");
    }
    if (2 == argc || argc == 4) {
        if (argc == 2) {
            fp = fopen(argv[1], "w");
        }
        else {
            fp = fopen(argv[3], "w");
        }

        if (!fp) {
            printf("Fail to open <output_file>, are you root?\n");
            fclose(fp);
            return 0;
        }

        if (is_big_endian())
            endian_flag = 0xff;
        else
            endian_flag = 0x0;

        fwrite(&endian_flag, 1, 1, fp);
    }

    if (argc == 3 || argc == 4) {
        clientSocket = socket(PF_INET, SOCK_DGRAM, 0);
        if (clientSocket == -1) {
            printf("Could not open Socket\n");
        }
        else {
            printf("Socket opened, sending to %s:%s\n", argv[1], argv[2]);
        }
        /*Configure settings in address struct*/
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(atoi(argv[2]));
        serverAddr.sin_addr.s_addr = inet_addr(argv[1]);
        memset(serverAddr.sin_zero, '\0', sizeof serverAddr.sin_zero);

        addr_size = sizeof serverAddr;
    }
    if (argc > 4) {
        printf(" Too many input arguments !\n");
        return 0;
    }

    fd = open_csi_device();
    if (fd < 0) {
        perror("Failed to open the device...");
        return errno;
    }

    printf("#Receiving data! Press Ctrl+C to quit!\n");

    quit = 0;
    total_msg_cnt = 0;
    data_send = 0;

    while (1) {
        if (1 == quit) {
            return 0;
            fclose(fp);
            close_csi_device(fd);
        }
        /* keep listening to the kernel and waiting for the csi report */
        cnt = read_csi_buf(buf_addr, fd, BUFSIZE);

        if (cnt) {
            total_msg_cnt += 1;

            /* fill the status struct with information about the rx packet */
            record_status(buf_addr, cnt, csi_status);

            /*
             * fill the payload buffer with the payload
             * fill the CSI matrix with the extracted CSI value
             */
            record_csi_payload(buf_addr, csi_status, data_buf, csi_matrix);

            /* Till now, we store the packet status in the struct csi_status
             * store the packet payload in the data buffer
             * store the csi matrix in the csi buffer
             * with all those data, we can build our own processing function!
             */
//            porcess_csi(data_buf, csi_status, csi_matrix);

            printf("Recv %dth msg with rate: 0x%02x | payload len: %d\n", total_msg_cnt, csi_status->rate,
                    csi_status->payload_len);
            printf("Timestamp %d\n", csi_status->tstamp);
//            printf("Channel: %d\n",csi_status->channel);
//            printf("Bandwidth: %d\n",csi_status->chanBW);
//            printf("Rate: %d\n",csi_status->rate);
//            printf("NR: %d\n",csi_status->nr);
//            printf("NC: %d\n",csi_status->nc);
//            printf("Tones: %d\n",csi_status->num_tones);
//            printf("Phyerr: %d\n",csi_status->phyerr);
//            printf("RSSI0: %d\n",csi_status->rssi);
//            printf("RSSI1: %d\n",csi_status->rssi_0);
//            printf("RSSI2: %d\n",csi_status->rssi_1);
//            printf("RSSI3: %d\n",csi_status->rssi_2);
//            printf("Payload Len: %d\n",csi_status->payload_len);
            printf("CSI Len: %d\n\n\n", csi_status->csi_len);

            buf_len = csi_status->buf_len;
            // Network Stuff
            if (argc == 3 || argc == 4) {
                printf("Sending Data\n");
                printf("buf len %d", buf_len);

                void* sendbuf = malloc(2 + buf_len);
                memcpy(sendbuf, &buf_len, 2);
                memcpy(sendbuf + 2, buf_addr, buf_len);
                data_send = sendto(clientSocket, sendbuf, (buf_len + 2), 0, (struct sockaddr*) &serverAddr, addr_size);
                printf("Total Data Send %d\n", data_send);
                free(sendbuf);
            }

//            /* log the received data for off-line processing */
//            if (log_flag){
//                fwrite(&buf_len,1,2,fp);
//                fwrite(buf_addr,1,buf_len,fp);
//            }
        }
    }

    fclose(fp);
    close_csi_device(fd);
    free(csi_status);

    return 0;
}
