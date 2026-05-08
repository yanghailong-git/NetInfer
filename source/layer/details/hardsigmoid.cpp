#include "hardsigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace net_infer {
using namespace activation;

// Constructor: initializes the hard sigmoid activation layer
// Hard sigmoid formula: max(0, min(1, (x + 3) / 6))
HardSigmoid::HardSigmoid()
    : ActivationLayer(ActivationType::kActivationHardSigmoid, "HardSigmoid") {}

// Forward pass: delegates to the base ActivationLayer's forward implementation
StatusCode HardSigmoid::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationLayer::Forward(inputs, outputs);
}

// Factory method: creates a HardSigmoid layer instance
StatusCode HardSigmoid::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                       std::shared_ptr<Layer<float>>& hardsigmoid_layer) {
  if (!op) {
    LOG(ERROR) << "The hardsigmoid operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  hardsigmoid_layer = std::make_shared<HardSigmoid>();
  return StatusCode::kSuccess;
}

// Register the hard sigmoid layer (corresponds to PyTorch's nn.Hardsigmoid)
LayerRegistererWrapper kHardSigmoidCreateInstance(HardSigmoid::CreateInstance, "nn.Hardsigmoid");

}  // namespace net_infer
