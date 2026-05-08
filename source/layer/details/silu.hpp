#ifndef NET_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief SiLU (Sigmoid Linear Unit) activation layer.
 *
 * Applies the element-wise SiLU function: f(x) = x * sigmoid(x).
 * Also known as Swish, commonly used in modern transformer architectures.
 */
class SiLULayer : public activation::ActivationLayer {
 public:
  explicit SiLULayer();

  /**
   * @brief Forward pass: apply SiLU activation to each input tensor.
   * @param inputs Input tensor batch.
   * @param outputs Output tensor batch.
   * @return Status code indicating success or failure.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief Factory method to create a SiLULayer from runtime operator parameters.
   * @param op Runtime operator (unused for stateless SiLU).
   * @param silu_layer Output layer pointer to be assigned.
   * @return Status code indicating success or failure.
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& silu_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_SILU_HPP_
