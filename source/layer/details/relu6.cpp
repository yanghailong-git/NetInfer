#include "relu6.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace net_infer {
using namespace activation;
StatusCode Relu6Layer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                               std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  return ActivationLayer::Forward(inputs, outputs);
}
StatusCode Relu6Layer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                      std::shared_ptr<Layer<float>>& relu_layer) {
  if (!op) {
    LOG(ERROR) << "The relu6 operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  relu_layer = std::make_shared<Relu6Layer>();
  return StatusCode::kSuccess;
}

Relu6Layer::Relu6Layer() : ActivationLayer(ActivationType::kActivationRelu6, "nn.ReLU6") {}

LayerRegistererWrapper kRelu6CreateInstance(Relu6Layer::CreateInstance, "nn.ReLU6");
}  // namespace net_infer
