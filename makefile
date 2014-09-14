CC = nvcc
CFLAGS = -O3 -Xcompiler -march=native 

all: word2vec 

word2vec : word2vec.cu
	$(CC) $< -o $@ $(CFLAGS) -arch=sm_35
