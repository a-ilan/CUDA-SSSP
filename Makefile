CC = /usr/local/cuda-7.5/bin/nvcc

sssp: *.cu 
	$(CC) -std=c++11 utils.cu entry_point.cu -O3 -arch=sm_30 -o sssp

run1:
	./sssp --bsize 1024 --bcount 2 --method bmf --input in.txt --output out.txt
run11:
	./sssp --bsize 1024 --bcount 2 --method bmf --input input/amazon0312.txt --output out.txt --sync outcore
run111:
	./sssp --bsize 1024 --bcount 2 --method bmf --input input/WebGoogle.txt --output out.txt


run2:
	./sssp --bsize 1024 --bcount 2 --method tpe --input in.txt --output out.txt
run22:
	./sssp --bsize 1024 --bcount 2 --method tpe --input input/amazon0312.txt --output out.txt --sync outcore
run222:
	./sssp --bsize 1024 --bcount 2 --method bmf --input input/LiveJournal.txt --output out.txt

clean:
	rm -f *.o sssp
