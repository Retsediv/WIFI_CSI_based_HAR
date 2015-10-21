OBJS = csi_fun.o main.o
CC = gcc
recv_csi: $(OBJS)
	$(CC) $(OBJS) -o recv_csi
csi_fun.o: csi_fun.c csi_fun.h
	$(CC) -c csi_fun.c -o csi_fun.o
main.o: main.c csi_fun.h
	$(CC) -c main.c -o main.o
clean: 
	rm -f *.o recv_csi

