#include "silu.hpp"
#include "activation.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"
#include "tick.hpp"

namespace net_infer {
using namespace activation;
StatusCode SiLULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationLayer::Forward(inputs, outputs);
}

StatusCode SiLULayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& silu_layer) {
  if (!op) {
    LOG(ERROR) << "The SiLU operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  silu_layer = std::make_shared<SiLULayer>();
  return StatusCode::kSuccess;
}

SiLULayer::SiLULayer() : ActivationLayer(ActivationType::kActivationSilu, "nn.SiLU") {}

LayerRegistererWrapper kSiluCreateInstance(SiLULayer::CreateInstance, "nn.SiLU");

}  // namespace net_infer
