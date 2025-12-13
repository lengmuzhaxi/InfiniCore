#ifndef __PIXEL_SHUFFLE_NVIDIA_CUH__
#define __PIXEL_SHUFFLE_NVIDIA_CUH__

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <stdio.h>

namespace op::pixel_shuffle::nvidia {

// --- 常量定义 ---
constexpr int BLOCK_SIZE = 256; // 元素级操作常用 256 或 512

// --- Kernel 实现 ---

template <typename T>
__global__ void pixel_shuffle_kernel(
    T *output,
    const T *input,
    size_t batch, size_t c_out, size_t h_out, size_t w_out,
    int64_t upscale_factor,
    size_t total_elements,
    // Strides
    ptrdiff_t out_s0, ptrdiff_t out_s1, ptrdiff_t out_s2, ptrdiff_t out_s3,
    ptrdiff_t inp_s0, ptrdiff_t inp_s1, ptrdiff_t inp_s2, ptrdiff_t inp_s3
) {
    // 线性索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // 1. 根据线性索引反解 Output 坐标 (n, c, h, w)
        // 假设内存布局是连续的逻辑计算，实际寻址用 stride
        size_t temp = idx;
        size_t w = temp % w_out; temp /= w_out;
        size_t h = temp % h_out; temp /= h_out;
        size_t c = temp % c_out; temp /= c_out;
        size_t n = temp; // 剩余部分为 batch

        // 2. 计算 Input 对应的坐标
        size_t h_in = h / upscale_factor;
        size_t w_in = w / upscale_factor;
        
        size_t offset_h = h % upscale_factor;
        size_t offset_w = w % upscale_factor;
        
        // Input Channel 计算
        size_t c_in = c * (upscale_factor * upscale_factor) + 
                      offset_h * upscale_factor + 
                      offset_w;

        // 3. 计算物理内存偏移量
        size_t out_offset = n * out_s0 + c * out_s1 + h * out_s2 + w * out_s3;
        size_t inp_offset = n * inp_s0 + c_in * inp_s1 + h_in * inp_s2 + w_in * inp_s3;

        // 4. 搬运数据
        output[out_offset] = input[inp_offset];
    }
}

// ==================================================================
// Launcher
// ==================================================================
template <typename T>
void launch_kernel(
    void *output, const void *input,
    size_t batch, size_t c_out, size_t h_out, size_t w_out,
    int64_t upscale_factor,
    // Strides
    ptrdiff_t out_s0, ptrdiff_t out_s1, ptrdiff_t out_s2, ptrdiff_t out_s3,
    ptrdiff_t inp_s0, ptrdiff_t inp_s1, ptrdiff_t inp_s2, ptrdiff_t inp_s3,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    size_t total_elements = batch * c_out * h_out * w_out;

    // 1D Grid 配置
    dim3 block(BLOCK_SIZE);
    dim3 grid((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    pixel_shuffle_kernel<T><<<grid, block, 0, cuda_stream>>>(
        out_ptr, in_ptr,
        batch, c_out, h_out, w_out,
        upscale_factor,
        total_elements,
        out_s0, out_s1, out_s2, out_s3,
        inp_s0, inp_s1, inp_s2, inp_s3
    );
}

} // namespace op::pixel_shuffle::nvidia

#endif // __PIXEL_SHUFFLE_NVIDIA_CUH__