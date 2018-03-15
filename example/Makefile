all:
	nvcc -g -O2 -x cu -arch=sm_61 -lineinfo -std=c++14 -Xcompiler="-fopenmp -Wall" -Xptxas="" -I.. main.cpp -o maingpu

cpu:
	g++ -g -O2 -std=c++14 -fopenmp -Wall -I.. main.cpp -o maincpu

clean:
	rm main
