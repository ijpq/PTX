debug:
	nvcc -arch=native -ptx -o test_mma.ptx mma.cu && \
	nvcc -DDEBUG_BUILD=1 -arch=native mma.cu && python3 reference.py && ./a.out	

release:
	nvcc -arch=native -ptx -o test_mma.ptx mma.cu && \
	nvcc -DDEBUG_BUILD=0 -arch=native mma.cu && python3 reference.py && ./a.out	

.DEFAULT_GOAL:=release
.PHONY: clean
clean:
	rm a.out && rm array.txt