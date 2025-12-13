#include "infinicore/ops/pixel_shuffle.hpp"
#include "../../utils.hpp"
#include <vector>

namespace infinicore::op {

// 1. 初始化 Dispatcher
common::OpDispatcher<PixelShuffle::schema> &PixelShuffle::dispatcher() {
    static common::OpDispatcher<PixelShuffle::schema> dispatcher_;
    return dispatcher_;
};

// 2. 类层面的 execute 方法
void PixelShuffle::execute(Tensor output, Tensor input, int upscale_factor) {
    // 切换上下文
    infinicore::context::setDevice(output->device());

    // 分发计算
    dispatcher().lookup(output->device().getType())(output, input, upscale_factor);
}

// 3. Out-of-place 接口
Tensor pixel_shuffle(Tensor input, int upscale_factor) {
    // Pixel Shuffle 会改变形状，必须计算新的 Shape
    const auto& in_shape = input->shape();
    
    // 假设输入格式为 NCHW (4D)
    // in_shape 类型通常是 std::vector<size_t>
    auto N = in_shape[0];
    auto C = in_shape[1];
    auto H = in_shape[2];
    auto W = in_shape[3];
    
    // 使用 size_t 避免有符号/无符号转换警告
    size_t uf = static_cast<size_t>(upscale_factor);
    size_t r2 = uf * uf;
    
    // [修复] 将类型 std::vector<int64_t> 修改为 Shape (即 std::vector<size_t>)
    Shape out_shape = {
        N, 
        C / r2, 
        H * uf, 
        W * uf
    };

    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    PixelShuffle::execute(output, input, upscale_factor);
    return output;
}

// 4. In-place 接口 (实际上是写入指定的 output tensor)
void pixel_shuffle_(Tensor output, Tensor input, int upscale_factor) {
    PixelShuffle::execute(output, input, upscale_factor);
}

} // namespace infinicore::op