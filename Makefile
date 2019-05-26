################ MAKEFILE TEMPLATE ################

# Author : Lucas Carpenter

# Usage : make target1

# What compiler are we using? (gcc, g++, nvcc, etc)
LINK = nvcc

# Name of our binary executable
OUT_FILE = test

# Any weird flags ( -O2/-O3/-Wno-deprecated-gpu-targets/-fopenmp/etc)
FLAGS = -Wno-deprecated-gpu-targets -O3 -Xcompiler -fopenmp -std=c++11 `pkg-config --cflags --libs opencv`


all: test

test: main.cu
	$(LINK) -o $(OUT_FILE) $(FLAGS) $^

clean: 
	rm -f *.o *~ test
