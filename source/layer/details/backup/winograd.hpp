#ifndef NET_INFER_WINOGRAD_HPP
#define NET_INFER_WINOGRAD_HPP
#include "data/tensor.hpp"
namespace net_infer {
// 使用 Winograd F(4x4, 3x3) 算法执行 3x3 stride-1 卷积。
// 对于大尺寸特征图，该方法比直接卷积显著更快。
void Convolution3x3s1(const std::shared_ptr<Tensor<float>>& input,
                      std::shared_ptr<Tensor<float>>& output, const std::vector<sftensor>& weights);
}  // namespace net_infer
#endif  // NET_INFER_WINOGRAD_HPP
