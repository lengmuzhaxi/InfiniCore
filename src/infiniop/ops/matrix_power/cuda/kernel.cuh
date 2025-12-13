#ifndef __MATRIX_POWER_NVIDIA_CUH__
#define __MATRIX_POWER_NVIDIA_CUH__

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <stdio.h>

namespace op::matrix_power::nvidia {

constexpr int BLOCK_SIZE = 16;

template <typename T>
__device__ __forceinline__ float to_float_acc(const T &x) {
    if constexpr (std::is_same_v<T, half>) return __half2float(x);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, nv_bfloat16>) return __bfloat162float(x);
#endif
    else return static_cast<float>(x);
}

template <typename T>
__device__ __forceinline__ T from_float_res(float x) {
    if constexpr (std::is_same_v<T, half>) return __float2half(x);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, nv_bfloat16>) return __float2bfloat16(x);
#endif
    else return static_cast<T>(x);
}

// 1. 单位矩阵 Kernel (用于 n=0)
template <typename T>
__global__ void identity_kernel(
    T *dst,
    size_t b, size_t m,
    ptrdiff_t dst_s0, ptrdiff_t dst_s1, ptrdiff_t dst_s2
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z;

    if (batch_idx < b && row < m && col < m) {
        size_t dst_idx = batch_idx * dst_s0 + row * dst_s1 + col * dst_s2;
        float val = (row == col) ? 1.0f : 0.0f;
        dst[dst_idx] = from_float_res<T>(val);
    }
}

// 2. 复制 Kernel
template <typename T>
__global__ void copy_kernel(
    T *dst, const T *src,
    size_t b, size_t m,
    ptrdiff_t dst_s0, ptrdiff_t dst_s1, ptrdiff_t dst_s2,
    ptrdiff_t src_s0, ptrdiff_t src_s1, ptrdiff_t src_s2
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z;

    if (batch_idx < b && row < m && col < m) {
        size_t src_idx = batch_idx * src_s0 + row * src_s1 + col * src_s2;
        size_t dst_idx = batch_idx * dst_s0 + row * dst_s1 + col * dst_s2;
        dst[dst_idx] = src[src_idx];
    }
}

// 3. 矩阵乘法 Kernel
template <typename T>
__global__ void matmul_kernel(
    T *C, const T *A, const T *B,
    size_t m,
    ptrdiff_t c_s0, ptrdiff_t c_s1, ptrdiff_t c_s2,
    ptrdiff_t a_s0, ptrdiff_t a_s1, ptrdiff_t a_s2,
    ptrdiff_t b_s0, ptrdiff_t b_s1, ptrdiff_t b_s2
) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int batch_idx = blockIdx.z;

    float acc = 0.0f;
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // 指针偏移到当前 Batch
    // 注意：这里的 s0 是 batch stride。
    // Kernel 内部通过 batch_idx * s0 计算偏移
    
    // 为了支持 batch 偏移，我们需要在循环内部计算完整偏移，或者在这里加上 batch 偏移
    // 简单起见，我们在 loop 里用 batch_idx * s0
    
    for (int k = 0; k < m; k += BLOCK_SIZE) {
        if (row < m && (k + threadIdx.x) < m) {
            size_t idx = batch_idx * a_s0 + row * a_s1 + (k + threadIdx.x) * a_s2;
            s_A[threadIdx.y][threadIdx.x] = to_float_acc(A[idx]);
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((k + threadIdx.y) < m && col < m) {
            size_t idx = batch_idx * b_s0 + (k + threadIdx.y) * b_s1 + col * b_s2;
            s_B[threadIdx.y][threadIdx.x] = to_float_acc(B[idx]);
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            acc += s_A[threadIdx.y][e] * s_B[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < m) {
        size_t idx = batch_idx * c_s0 + row * c_s1 + col * c_s2;
        C[idx] = from_float_res<T>(acc);
    }
}

// Launcher
template <typename T>
void launch_kernel(
    void *output, const void *input, void *workspace,
    size_t b, size_t m, int64_t n,
    ptrdiff_t out_s0, ptrdiff_t out_s1, ptrdiff_t out_s2,
    ptrdiff_t inp_s0, ptrdiff_t inp_s1, ptrdiff_t inp_s2,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto work_ptr = reinterpret_cast<T *>(workspace);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 
        (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
        b
    );

    // Workspace 是紧凑的
    ptrdiff_t ws_s2 = 1;
    ptrdiff_t ws_s1 = m;
    ptrdiff_t ws_s0 = m * m;

    // Case 0: n = 0 (Output = Identity)
    if (n == 0) {
        identity_kernel<T><<<grid, block, 0, cuda_stream>>>(
            out_ptr, b, m, out_s0, out_s1, out_s2
        );
        return;
    }

    // Case 1: n = 1 (Output = Input)
    // 必须使用 copy_kernel 处理 stride
    if (n == 1) {
        copy_kernel<T><<<grid, block, 0, cuda_stream>>>(
            out_ptr, in_ptr, b, m,
            out_s0, out_s1, out_s2,
            inp_s0, inp_s1, inp_s2
        );
        return;
    }

    // Case 2: n >= 2
    // 1. Init: Output = Input (Copy)
    copy_kernel<T><<<grid, block, 0, cuda_stream>>>(
        out_ptr, in_ptr, b, m,
        out_s0, out_s1, out_s2,
        inp_s0, inp_s1, inp_s2
    );

    // 2. Loop
    for (int i = 1; i < n; ++i) {
        // Step A: Backup Output -> Workspace (Workspace is compact)
        copy_kernel<T><<<grid, block, 0, cuda_stream>>>(
            work_ptr, out_ptr, b, m,
            ws_s0, ws_s1, ws_s2,
            out_s0, out_s1, out_s2
        );

        // Step B: Output = Workspace * Input
        // Workspace(A) is compact, Input(B) uses input strides, Output(C) uses output strides
        matmul_kernel<T><<<grid, block, 0, cuda_stream>>>(
            out_ptr, work_ptr, in_ptr,
            m,
            out_s0, out_s1, out_s2, // C
            ws_s0, ws_s1, ws_s2,    // A
            inp_s0, inp_s1, inp_s2  // B
        );
    }
}

} // namespace op::matrix_power::nvidia

#endif // __MATRIX_POWER_NVIDIA_CUH__