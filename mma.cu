#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

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
    __shared__ int32_t dst_shared[8*8];

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

    // while loading m8n8 .x1 matrix, thread0~thread7 get 8 rows address of the matrix respectively.
    unsigned row_src_a = cutlass_get_smem_pointer(src_a + 16 *(threadIdx.x % 8));

    unsigned col_src_b = cutlass_get_smem_pointer(src_b + 16 * (threadIdx.x % 8));
    unsigned smem_dst = cutlass_get_smem_pointer(dst_shared);

    int32_t r_a, r_b;
    // smem -> reg
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {%0}, [%1];"
                 : "=r"(r_a)
                 : "r"(row_src_a));
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {%0}, [%1];"
                 : "=r"(r_b)
                 : "r"(col_src_b));
    /* DEBUG
    printf("tid: %d, val: %d, %d, %d, %d", threadIdx.x, (int8_t)((r_b) & 0xFF), (int8_t)((r_b >> 8) & 0xFF), (int8_t)((r_b >> 16) & 0xFF), (int8_t)((r_b >> 24) & 0xFF));
    */
    __syncthreads();

    int32_t r_c_0 = 0, r_c_1=0, r_d_0, r_d_1;
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=r"(r_d_0), "=r"(r_d_1)
                 : "r"(r_a), "r"(r_b), "r"(r_c_0), "r"(r_c_1)
    );
    // reg -> smem
    /* DEBUG
    printf("tid: %d, d0: %d, d1: %d\n", threadIdx.x, r_d_0, r_d_1);
    */
    auto idx_dst = 2 * threadIdx.x;
    dst_shared[idx_dst] = r_d_0;
    dst_shared[idx_dst + 1] = r_d_1;

    __syncthreads();

    int32_t *dst_ptr = (int32_t*)dst;
    dst_ptr[idx_dst] = dst_shared[idx_dst];
    dst_ptr[idx_dst + 1] = dst_shared[idx_dst + 1];

    /* DEBUG
    // printf("tid: %d, %d, ", threadIdx.x, (r_a >> (0 * 8)) & 0xFF);
    // printf("tid: %d, %d, ", threadIdx.x, (r_a >> (1 * 8)) & 0xFF);
    // printf("tid: %d, %d, ", threadIdx.x, (r_a >> (2 * 8)) & 0xFF);
    // printf("tid: %d, %d, ", threadIdx.x, (r_a >> (3 * 8)) & 0xFF);
    __syncthreads();
    */

    // smem -> gmem
    return;
}

void get_reference(std::vector<int32_t>& data) {

    std::ifstream file("array.txt");
    int value;

    // Check if the file opened successfully
    if (!file) {
        std::cerr << "Unable to open file!" << std::endl;
        return;
    }

    while (file >> value) {
        data.push_back(value);
    }
    file.close();

    // Print the vector to verify the contents
    for (int val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

}

int main() {

    // cudamalloc
    int8_t *dst_dev_ptr = nullptr, *src_A_dev_ptr = nullptr,
           *src_B_dev_ptr = nullptr;
    cudaMalloc(&dst_dev_ptr, m8n8k16_dst_size * sizeof(int32_t));
    cudaMalloc(&src_A_dev_ptr, m8n8k16_src_size);
    cudaMalloc(&src_B_dev_ptr, m8n8k16_src_size);

    // host mem alloc
    std::vector<int8_t> src_A(m8n8k16_src_size), src_B(m8n8k16_src_size);
    std::vector<int32_t> dst(m8n8k16_dst_size);

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
                               dst_dev_ptr); // static alloc shared mem
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
        printf("err: %s\n", cudaGetErrorString(err));

    // cuda memcpy
    cudaMemcpy(dst.data(), dst_dev_ptr, m8n8k16_dst_size * sizeof(int32_t),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
        printf("err: %s\n", cudaGetErrorString(err));

    std::cout << "result, dst:" << std::endl;
    for (auto i : dst) {
        printf("%d, ", i);
    }
    std::cout << std::endl;

    std::vector<int32_t> reference;
    get_reference(reference);
    

    assert(reference.size() == dst.size());
    for (size_t i =0 ; i < reference.size(); i++) {
        assert(reference[i] == dst[i]);
    }
    std::cout << "passed" << std::endl;
    return 0;
}