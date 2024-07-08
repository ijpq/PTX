#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define m8n8k16_src_size (16 * 8)
#define m8n8k16_dst_size (8 * 8)

inline __device__ unsigned cutlass_get_smem_pointer(void *ptr) {

    // We prefer to use the new CVTA intrinsics if they are available, otherwise
    // we will fall back to the previous internal intrinsics if they are
    // available.
    //
    // This NVVM intrinsic converts an address in shared memory to a plain
    // unsigned integer. This is necessary to pass to shared memory instructions
    // in inline PTX.
    //
    // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only
    // available in 10.2].
    //
    //__device__ size_t __cvta_generic_to_shared(void* ptr);

    /// CUTLASS helper to get SMEM pointer
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__global__ void kernel(int8_t *src_A, int8_t *src_B, void *dst) {

    __shared__ int8_t src_a[128];
    __shared__ int8_t src_b[128];

    // gmem -> smem
    int idx = 4 * threadIdx.x;
    src_a[idx] = src_A[idx];
    src_a[idx + 1] = src_A[idx + 1];
    src_a[idx + 2] = src_A[idx + 2];
    src_a[idx + 3] = src_A[idx + 3];

    src_b[idx] = src_B[idx];
    src_b[idx + 1] = src_B[idx + 1];
    src_b[idx + 2] = src_B[idx + 2];
    src_b[idx + 3] = src_B[idx + 3];

    __syncthreads();

    // impl PTX mma
    unsigned row_src_a = cutlass_get_smem_pointer(src_a + 16 *(threadIdx.x % 8));
    unsigned smem_src_a = cutlass_get_smem_pointer(src_a);
    unsigned smem_src_b = cutlass_get_smem_pointer(src_b);

    int32_t r_a, r_b;
    // smem -> reg
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {%0}, [%1];"
                 : "=r"(r_a)
                 : "r"(row_src_a));
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {%0}, [%1];"
                 : "=r"(r_b)
                 : "r"(smem_src_b));

    // reg -> smem
    src_b[idx] = (r_a >> (0 * 8)) & 0xFF;
    src_b[idx + 1] = (r_a >> (1 * 8)) & 0xFF;
    src_b[idx + 2] = (r_a >> (2 * 8)) & 0xFF;
    src_b[idx + 3] = (r_a >> (3 * 8)) & 0xFF;

    // printf("tid: %d, %d, ", threadIdx.x, (r_a >> (0 * 8)) & 0xFF);
    // printf("tid: %d, %d, ", threadIdx.x, (r_a >> (1 * 8)) & 0xFF);
    // printf("tid: %d, %d, ", threadIdx.x, (r_a >> (2 * 8)) & 0xFF);
    // printf("tid: %d, %d, ", threadIdx.x, (r_a >> (3 * 8)) & 0xFF);
    __syncthreads();

    // smem -> gmem
    src_A[idx] = src_a[idx];
    src_A[idx + 1] = src_a[idx + 1];
    src_A[idx + 2] = src_a[idx + 2];
    src_A[idx + 3] = src_a[idx + 3];

    src_B[idx] = src_b[idx];
    src_B[idx + 1] = src_b[idx + 1];
    src_B[idx + 2] = src_b[idx + 2];
    src_B[idx + 3] = src_b[idx + 3];

    return;
}

int main() {

    // cudamalloc
    int8_t *dst_dev_ptr = nullptr, *src_A_dev_ptr = nullptr,
           *src_B_dev_ptr = nullptr;
    cudaMalloc(&dst_dev_ptr, m8n8k16_dst_size);
    cudaMalloc(&src_A_dev_ptr, m8n8k16_src_size);
    cudaMalloc(&src_B_dev_ptr, m8n8k16_src_size);

    // host mem alloc
    std::vector<int8_t> src_A(m8n8k16_src_size), src_B(m8n8k16_src_size);
    std::vector<int8_t> dst(m8n8k16_dst_size);

    for (int i = 0; i < m8n8k16_src_size; ++i) {
        src_A[i] = i;
        src_B[i] = -1 * (i+1);
    }
    std::cout << "initial status src A:" << std::endl;
    for (auto i : src_A) {
        printf("%d, ", i);
    }
    std::cout << std::endl;
    std::cout << "initial status src B:" << std::endl;
    for (auto i : src_B) {
        printf("%d, ", i);
    }
    std::cout << std::endl;

    // cuda memcpy
    cudaMemcpy(src_A_dev_ptr, src_A.data(), m8n8k16_src_size,
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(src_B_dev_ptr, src_B.data(), m8n8k16_src_size,
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
        printf("err: %s\n", cudaGetErrorString(err));

    // launch kernel
    dim3 grids{1, 1, 1};
    dim3 threads{32, 1, 1};

    kernel<<<grids, threads>>>(src_A_dev_ptr, src_B_dev_ptr,
                               src_B_dev_ptr); // static alloc shared mem
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
        printf("err: %s\n", cudaGetErrorString(err));

    // cuda memcpy
    // cudaMemcpy(dst.data(), dst_dev_ptr, m8n8k16_dst_size,
    //            cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(src_B.data(), src_B_dev_ptr, m8n8k16_src_size,
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
        printf("err: %s\n", cudaGetErrorString(err));

    std::cout << "result, src_B:" << std::endl;
    for (auto i : src_B) {
        printf("%d, ", i);
    }
    std::cout << std::endl;

    return 0;
}