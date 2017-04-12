CC = /usr/local/cuda-7.5/bin/nvcc

sssp: *.cu 
	$(CC) -std=c++11 utils.cu entry_point.cu -O3 -arch=sm_30 -o sssp

clean:
	rm -f *.o sssp

tar:
	tar -cvzf sssp_gpu.tgz *.cu *.cuh *.pdf README Makefile

untar:
	tar -xvzf sssp_gpu.tgz

