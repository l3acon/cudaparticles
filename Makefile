# Load CUDA using the following command
# module load cuda
#

CUDADIR = /usr/local/cuda
CUDAINC = -I$(CUDADIR)/include
NVCCLIBS = -L$(CUDADIR)/lib64 -Xlinker -rpath -Xlinker /usr/local/cuda/lib64

CC = $(CUDADIR)/bin/nvcc
CFLAGS = -O3 -arch=compute_20 -code=sm_20
NVCCFLAGS = -O3 $(CUDAINC) -arch=compute_20 -code=sm_20
LIBS = 

TARGETS = serial gpu autograder

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o
	$(CC) $(NVCCFLAGS) -o $@ gpu.o common.o $(NVCCLIBS)
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o

serial.o: serial.cu common.h
	$(CC) -c $(CFLAGS) serial.cu
autograder.o: autograder.cu common.h
	$(CC) -c $(CFLAGS) autograder.cu
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
