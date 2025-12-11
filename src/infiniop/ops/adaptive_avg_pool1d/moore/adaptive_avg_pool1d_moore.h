#ifndef __ADAPTIVE_AVG_POOL1D_MOORE_API_H__
#define __ADAPTIVE_AVG_POOL1D_MOORE_API_H__

// 引入上层定义的 Descriptor 宏和基础类
#include "../adaptive_avg_pool1d.h"

// 使用 adaptive_avg_pool1d.h 中定义的 DESCRIPTOR 宏
// 这将自动生成 op::adaptive_avg_pool1d::moore::Descriptor 类定义
DESCRIPTOR(moore)

#endif // __ADAPTIVE_AVG_POOL1D_MOORE_API_H__