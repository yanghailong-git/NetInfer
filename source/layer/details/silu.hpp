#ifndef NET_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief SiLU（Sigmoid Linear Unit）激活层。
 *
 * 逐元素应用 SiLU 函数：f(x) = x * sigmoid(x)。
 * 也称为 Swish，常用于现代 Transformer 架构。
 */
class SiLULayer : public activation::ActivationLayer {
 public:
  explicit SiLULayer();

  /**
   * @brief 前向传播：对每个输入张量应用 SiLU 激活。
   * @param inputs 输入张量批次。
   * @param outputs 输出张量批次。
   * @return 表示成功或失败的状态码。
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief 工厂方法，根据运行时算子参数创建 SiLULayer。
   * @param op 运行时算子（对无状态的 SiLU 不使用）。
   * @param silu_layer 待赋值的输出层指针。
   * @return 表示成功或失败的状态码。
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& silu_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
