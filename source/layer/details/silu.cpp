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
  // 委托给基类 ActivationLayer 执行实际的 SiLU 计算。
  return ActivationLayer::Forward(inputs, outputs);
}

StatusCode SiLULayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& silu_layer) {
  if (!op) {
    LOG(ERROR) << "The SiLU operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  // SiLU 无参数；直接实例化该层。
  silu_layer = std::make_shared<SiLULayer>();
  return StatusCode::kSuccess;
}

SiLULayer::SiLULayer() : ActivationLayer(ActivationType::kActivationSilu, "nn.SiLU") {}

LayerRegistererWrapper kSiluCreateInstance(SiLULayer::CreateInstance, "nn.SiLU");

}  // namespace net_infer
