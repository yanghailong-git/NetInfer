#ifndef NET_INFER_INCLUDE_MATH_ARMA_SSE
#define NET_INFER_INCLUDE_MATH_ARMA_SSE
#include "activation.hpp"
namespace net_infer {
namespace activation {
// Returns a pointer to the SIMD-accelerated activation function for the requested type.
ActivationFunc ApplySSEActivation(ActivationType act_type);

}  // namespace activation
}  // namespace net_infer
#endif  // NET_INFER_INCLUDE_MATH_ARMA_SSE
