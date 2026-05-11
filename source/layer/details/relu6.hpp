#ifndef NET_INFER_SOURCE_LAYER_BINOCULAR_RELU6_HPP_
#define NET_INFER_SOURCE_LAYER_BINOCULAR_RELU6_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief ReLU6 激活层。
 *
 * 逐元素应用 ReLU6 函数：f(x) = min(max(0, x), 6)。
 * 这是一种裁剪后的 ReLU，将输出上限设为 6，常用于
 * 移动端/高效网络，如 MobileNet。
 */
class Relu6Layer : public activation::ActivationLayer  {
 public:
  explicit Relu6Layer();

  /**
   * @brief 前向传播：对每个输入张量应用 ReLU6 激活。
   * @param inputs 输入张量批次。
   * @param outputs 输出张量批次。
   * @return 表示成功或失败的状态码。
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief 工厂方法，根据运行时算子参数创建 Relu6Layer。
   * @param op 运行时算子（对无状态的 ReLU6 不使用）。
   * @param relu_layer 待赋值的输出层指针。
   * @return 表示成功或失败的状态码。
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& relu_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
