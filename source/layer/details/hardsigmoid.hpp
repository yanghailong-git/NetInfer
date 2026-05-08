#ifndef NET_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

// Hard Sigmoid activation layer
// Computes the hard sigmoid activation: max(0, min(1, (x + 3) / 6))
// This is a piece-wise linear approximation of the standard sigmoid function
// Corresponds to PyTorch's nn.Hardsigmoid
class HardSigmoid : public activation::ActivationLayer {
 public:
  // Constructor: initializes the hard sigmoid activation layer
  explicit HardSigmoid();

  // Forward pass: applies hard sigmoid activation to each element of the input tensors
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // Factory method: creates a HardSigmoid layer from a RuntimeOperator
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& hardsigmoid_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
