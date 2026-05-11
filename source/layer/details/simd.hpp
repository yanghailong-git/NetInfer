#ifndef NET_INFER_INCLUDE_MATH_ARMA_SSE
#define NET_INFER_INCLUDE_MATH_ARMA_SSE
#include "activation.hpp"
namespace net_infer {
namespace activation {
// 返回指定类型的 SIMD 加速激活函数指针。
ActivationFunc ApplySSEActivation(ActivationType act_type);

}  // namespace activation
}  // namespace net_infer
#endif  // NET_INFER_INCLUDE_MATH_ARMA_SSE
