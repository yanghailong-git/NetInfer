#include "hardswish.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace net_infer {
using namespace activation;

// Constructor: initializes the hard swish activation layer
// Hard swish formula: x * max(0, min(1, (x + 3) / 6))
HardSwishLayer::HardSwishLayer()
    : ActivationLayer(ActivationType::kActivationHardSwish, "HardSwish") {}

// Forward pass: delegates to the base ActivationLayer's forward implementation
StatusCode HardSwishLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationLayer::Forward(inputs, outputs);
}

// Factory method: creates a HardSwishLayer instance
StatusCode HardSwishLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                          std::shared_ptr<Layer<float>>& hardswish_layer) {
  if (!op) {
    LOG(ERROR) << "The hardswish operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  hardswish_layer = std::make_shared<HardSwishLayer>();
  return StatusCode::kSuccess;
}

// Register the hard swish layer (corresponds to PyTorch's nn.Hardswish)
LayerRegistererWrapper kHardSwishCreateInstance(HardSwishLayer::CreateInstance, "nn.Hardswish");

}  // namespace net_infer
