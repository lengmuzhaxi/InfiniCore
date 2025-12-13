#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class PixelShuffle {
public:
    // 参数: output, input, upscale_factor
    using schema = void (*)(Tensor, Tensor, int);
    static void execute(Tensor output, Tensor input, int upscale_factor);
    
    static common::OpDispatcher<schema> &dispatcher();
};

// Out-of-place: pixel_shuffle(input, upscale_factor) -> Tensor
Tensor pixel_shuffle(Tensor input, int upscale_factor);

// In-place-like (writing to specific output): pixel_shuffle_(output, input, upscale_factor)
void pixel_shuffle_(Tensor output, Tensor input, int upscale_factor);

} // namespace infinicore::op