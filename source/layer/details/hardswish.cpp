#include "hardswish.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace net_infer {
using namespace activation;

// 构造函数：初始化 Hard Swish 激活层
// Hard Swish 公式：x * max(0, min(1, (x + 3) / 6))
HardSwishLayer::HardSwishLayer()
    : ActivationLayer(ActivationType::kActivationHardSwish, "HardSwish") {}

// 前向传播：委托给基类 ActivationLayer 的前向实现
StatusCode HardSwishLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationLayer::Forward(inputs, outputs);
}

// 工厂方法：创建 HardSwishLayer 实例
StatusCode HardSwishLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                          std::shared_ptr<Layer<float>>& hardswish_layer) {
  if (!op) {
    LOG(ERROR) << "The hardswish operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }
  hardswish_layer = std::make_shared<HardSwishLayer>();
  return StatusCode::kSuccess;
}

// 注册 Hard Swish 层（对应 PyTorch 的 nn.Hardswish）
LayerRegistererWrapper kHardSwishCreateInstance(HardSwishLayer::CreateInstance, "nn.Hardswish");

}  // namespace net_infer
