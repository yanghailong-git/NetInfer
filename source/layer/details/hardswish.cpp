#include "hardswish.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace net_infer {
using namespace activation;
HardSwishLayer::HardSwishLayer()
    : ActivationLayer(ActivationType::kActivationHardSwish, "HardSwish") {}

StatusCode HardSwishLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationLayer::Forward(inputs, outputs);
}

StatusCode HardSwishLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                          std::shared_ptr<Layer<float>>& hardswish_layer) {
  if (!op) {
    LOG(ERROR) << "The hardswish operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  hardswish_layer = std::make_shared<HardSwishLayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kHardSwishCreateInstance(HardSwishLayer::CreateInstance, "nn.Hardswish");

}  // namespace net_infer