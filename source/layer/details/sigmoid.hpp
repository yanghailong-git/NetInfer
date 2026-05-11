#ifndef NET_INFER_SOURCE_LAYER_SIGMOID_HPP_
#define NET_INFER_SOURCE_LAYER_SIGMOID_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief Sigmoid 激活层。
 *
 * 逐元素应用 Sigmoid 函数：f(x) = 1 / (1 + exp(-x))。
 * 实际计算委托给基类 ActivationLayer。
 */
class SigmoidLayer : public activation::ActivationLayer {
 public:
  explicit SigmoidLayer();

  /**
   * @brief 前向传播：对每个输入张量应用 Sigmoid 激活。
   * @param inputs 输入张量批次。
   * @param outputs 输出张量批次。
   * @return 表示成功或失败的状态码。
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief 工厂方法，根据运行时算子参数创建 SigmoidLayer。
   * @param op 运行时算子（对无状态的 Sigmoid 不使用）。
   * @param sigmoid_layer 待赋值的输出层指针。
   * @return 表示成功或失败的状态码。
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& sigmoid_layer);
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_SIGMOID_HPP_
