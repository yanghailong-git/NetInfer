#include "hardsigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace net_infer {
using namespace activation;

// 构造函数：初始化 Hard Sigmoid 激活层
// Hard Sigmoid 公式：max(0, min(1, (x + 3) / 6))
HardSigmoid::HardSigmoid()
    : ActivationLayer(ActivationType::kActivationHardSigmoid, "HardSigmoid") {}

// 前向传播：委托给基类 ActivationLayer 的前向实现
StatusCode HardSigmoid::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationLayer::Forward(inputs, outputs);
}

// 工厂方法：创建 HardSigmoid 层实例
StatusCode HardSigmoid::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                       std::shared_ptr<Layer<float>>& hardsigmoid_layer) {
  if (!op) {
    LOG(ERROR) << "The hardsigmoid operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  hardsigmoid_layer = std::make_shared<HardSigmoid>();
  return StatusCode::kSuccess;
}

// 注册 Hard Sigmoid 层（对应 PyTorch 的 nn.Hardsigmoid）
LayerRegistererWrapper kHardSigmoidCreateInstance(HardSigmoid::CreateInstance, "nn.Hardsigmoid");

}  // namespace net_infer
