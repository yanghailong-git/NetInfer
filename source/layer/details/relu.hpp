#ifndef NET_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#define NET_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief ReLU (Rectified Linear Unit) activation layer.
 *
 * Applies the element-wise ReLU function: f(x) = max(0, x).
 * Delegates the actual computation to the base ActivationLayer.
 */
class ReluLayer : public activation::ActivationLayer {
 public:
  explicit ReluLayer();

  /**
   * @brief Forward pass: apply ReLU activation to each input tensor.
   * @param inputs Input tensor batch.
   * @param outputs Output tensor batch.
   * @return Status code indicating success or failure.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief Factory method to create a ReluLayer from runtime operator parameters.
   * @param op Runtime operator (unused for stateless ReLU).
   * @param relu_layer Output layer pointer to be assigned.
   * @return Status code indicating success or failure.
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& relu_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
