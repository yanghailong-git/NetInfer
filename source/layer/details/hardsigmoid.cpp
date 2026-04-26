#include "hardsigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace net_infer {
using namespace activation;
HardSigmoid::HardSigmoid()
    : ActivationLayer(ActivationType::kActivationHardSigmoid, "HardSigmoid") {}

StatusCode HardSigmoid::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationLayer::Forward(inputs, outputs);
}

StatusCode HardSigmoid::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                       std::shared_ptr<Layer<float>>& hardsigmoid_layer) {
  if (!op) {
    LOG(ERROR) << "The hardsigmoid operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  hardsigmoid_layer = std::make_shared<HardSigmoid>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kHardSigmoidCreateInstance(HardSigmoid::CreateInstance, "nn.Hardsigmoid");

}  // namespace net_infer
