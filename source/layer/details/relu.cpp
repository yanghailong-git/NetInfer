#include "relu.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace net_infer {
using namespace activation;
ReluLayer::ReluLayer() : ActivationLayer(ActivationType::kActivationRelu, "nn.ReLU") {}

StatusCode ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  return ActivationLayer::Forward(inputs, outputs);
}

StatusCode ReluLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& relu_layer) {
  if (!op) {
    LOG(ERROR) << "The relu operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  relu_layer = std::make_shared<ReluLayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kReluCreateInstance(ReluLayer::CreateInstance, "nn.ReLU");
}  // namespace net_infer
