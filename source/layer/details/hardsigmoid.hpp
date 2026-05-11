#ifndef NET_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

// Hard Sigmoid 激活层
// 计算 hard sigmoid 激活：max(0, min(1, (x + 3) / 6))
// 这是标准 sigmoid 函数的分段线性近似
// 对应 PyTorch 的 nn.Hardsigmoid
class HardSigmoid : public activation::ActivationLayer {
 public:
  // 构造函数：初始化 hard sigmoid 激活层
  explicit HardSigmoid();

  // 前向传播：对输入张量的每个元素应用 hard sigmoid 激活
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // 工厂方法：从 RuntimeOperator 创建 HardSigmoid 层
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& hardsigmoid_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
