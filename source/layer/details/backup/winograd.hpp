#ifndef NET_INFER_WINOGRAD_HPP
#define NET_INFER_WINOGRAD_HPP
#include "data/tensor.hpp"
namespace net_infer {
// Performs a 3x3 stride-1 convolution using the Winograd F(4x4, 3x3) algorithm.
// This can be significantly faster than direct convolution for large feature maps.
void Convolution3x3s1(const std::shared_ptr<Tensor<float>>& input,
                      std::shared_ptr<Tensor<float>>& output, const std::vector<sftensor>& weights);
}  // namespace net_infer
#endif  // NET_INFER_WINOGRAD_HPP
