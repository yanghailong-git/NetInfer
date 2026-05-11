#ifndef NET_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
#include "layer/abstract/non_param_layer.hpp"
#include "activation.hpp"

namespace net_infer {

// Hard Swish 激活层
// 计算 hard swish 激活：x * max(0, min(1, (x + 3) / 6))
// 将 hard sigmoid 与输入相乘，以获得更好的移动端性能
// 对应 PyTorch 的 nn.Hardswish
class HardSwishLayer : public activation::ActivationLayer {
 public:
  // 构造函数：初始化 hard swish 激活层
  explicit HardSwishLayer();

  // 前向传播：对输入张量的每个元素应用 hard swish 激活
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // 工厂方法：从 RuntimeOperator 创建 HardSwishLayer
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& hardswish_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
