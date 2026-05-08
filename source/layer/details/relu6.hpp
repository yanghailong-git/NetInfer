#ifndef NET_INFER_SOURCE_LAYER_BINOCULAR_RELU6_HPP_
#define NET_INFER_SOURCE_LAYER_BINOCULAR_RELU6_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief ReLU6 activation layer.
 *
 * Applies the element-wise ReLU6 function: f(x) = min(max(0, x), 6).
 * This is a clipped ReLU that caps the output at 6, commonly used in
 * mobile/efficient networks such as MobileNet.
 */
class Relu6Layer : public activation::ActivationLayer  {
 public:
  explicit Relu6Layer();

  /**
   * @brief Forward pass: apply ReLU6 activation to each input tensor.
   * @param inputs Input tensor batch.
   * @param outputs Output tensor batch.
   * @return Status code indicating success or failure.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief Factory method to create a Relu6Layer from runtime operator parameters.
   * @param op Runtime operator (unused for stateless ReLU6).
   * @param relu_layer Output layer pointer to be assigned.
   * @return Status code indicating success or failure.
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& relu_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
