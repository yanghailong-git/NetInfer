#include "sigmoid.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace net_infer {
using namespace activation;
SigmoidLayer::SigmoidLayer() : ActivationLayer(ActivationType::kActivationSigmoid, "nn.Sigmoid") {}

StatusCode SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationLayer::Forward(inputs, outputs);
}

StatusCode SigmoidLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& sigmoid_layer) {
  if (!op) {
    LOG(ERROR) << "The sigmoid operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  sigmoid_layer = std::make_shared<SigmoidLayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kSigmoidCreateInstance(SigmoidLayer::CreateInstance, "nn.Sigmoid");
}  // namespace net_infer