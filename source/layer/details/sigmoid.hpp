#ifndef NET_INFER_SOURCE_LAYER_SIGMOID_HPP_
#define NET_INFER_SOURCE_LAYER_SIGMOID_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief Sigmoid activation layer.
 *
 * Applies the element-wise sigmoid function: f(x) = 1 / (1 + exp(-x)).
 * Delegates the actual computation to the base ActivationLayer.
 */
class SigmoidLayer : public activation::ActivationLayer {
 public:
  explicit SigmoidLayer();

  /**
   * @brief Forward pass: apply sigmoid activation to each input tensor.
   * @param inputs Input tensor batch.
   * @param outputs Output tensor batch.
   * @return Status code indicating success or failure.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief Factory method to create a SigmoidLayer from runtime operator parameters.
   * @param op Runtime operator (unused for stateless Sigmoid).
   * @param sigmoid_layer Output layer pointer to be assigned.
   * @return Status code indicating success or failure.
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& sigmoid_layer);
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_SIGMOID_HPP_
