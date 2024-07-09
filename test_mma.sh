nvcc -arch=native -ptx -o test_mma.ptx mma.cu && \
nvcc -arch=native mma.cu && python3 reference.py && ./a.out
