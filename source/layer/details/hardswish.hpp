#ifndef NET_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
#include "layer/abstract/non_param_layer.hpp"
#include "activation.hpp"

namespace net_infer {

// Hard Swish activation layer
// Computes the hard swish activation: x * max(0, min(1, (x + 3) / 6))
// Combines the hard sigmoid with a multiplication by the input for better mobile performance
// Corresponds to PyTorch's nn.Hardswish
class HardSwishLayer : public activation::ActivationLayer {
 public:
  // Constructor: initializes the hard swish activation layer
  explicit HardSwishLayer();

  // Forward pass: applies hard swish activation to each element of the input tensors
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // Factory method: creates a HardSwishLayer from a RuntimeOperator
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& hardswish_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
