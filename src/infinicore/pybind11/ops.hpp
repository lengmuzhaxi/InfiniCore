#pragma once

#include <pybind11/pybind11.h>
#include "ops/acos.hpp"
#include "ops/adaptive_avg_pool1d.hpp"
#include "ops/addbmm.hpp"
#include "ops/add.hpp"
#include "ops/affine_grid.hpp" 
#include "ops/attention.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/floor.hpp"
#include "ops/floor_divide.hpp"
#include "ops/float_power.hpp"
#include "ops/flipud.hpp"
#include "ops/hypot.hpp"
#include "ops/index_add.hpp"
#include "ops/kthvalue.hpp"
#include "ops/index_copy.hpp"
#include "ops/linear.hpp"
#include "ops/lerp.hpp"
#include "ops/ldexp.hpp"
#include "ops/logcumsumexp.hpp"
#include "ops/logical_and.hpp" 
#include "ops/logical_not.hpp" 
#include "ops/matmul.hpp"
#include "ops/multi_margin_loss.hpp"
#include "ops/margin_ranking_loss.hpp"
#include "ops/triplet_margin_loss.hpp"
#include "ops/mul.hpp"
#include "ops/paged_attention.hpp"
#include "ops/paged_caching.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/silu.hpp"
#include "ops/scatter.hpp"
#include "ops/smooth_l1_loss.hpp"
#include "ops/swiglu.hpp"
#include "ops/take.hpp"
#include "ops/vander.hpp"
#include "ops/unfold.hpp"
#include "ops/upsample_bilinear.hpp"
#include "ops/pairwise_distance.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_acos(m);
    bind_adaptive_avg_pool1d(m);
    bind_add(m);
    bind_affine_grid(m); 
    bind_addbmm(m);
    bind_attention(m);
    bind_causal_softmax(m);
    bind_random_sample(m);
    bind_floor(m);
    bind_floor_divide(m);
    bind_float_power(m);
    bind_flipud(m);
    bind_hypot(m);
    bind_index_add(m);
    bind_index_copy(m);
    bind_kthvalue(m);
    bind_ldexp(m);
    bind_lerp(m);
    bind_linear(m);
    bind_logcumsumexp(m);
    bind_logical_and(m); 
    bind_logical_not(m); 
    bind_matmul(m);
    bind_mul(m);
    bind_multi_margin_loss(m);
    bind_margin_ranking_loss(m);
    bind_triplet_margin_loss(m);
    bind_pairwise_distance(m);
    bind_paged_attention(m);
    bind_paged_caching(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_silu(m);
    bind_scatter(m);
    bind_smooth_l1_loss(m);
    bind_swiglu(m);
    bind_take(m);
    bind_vander(m);
    bind_unfold(m);
    bind_upsample_bilinear(m);
    bind_rope(m);
    bind_embedding(m);
}

} // namespace infinicore::ops