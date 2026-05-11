#ifndef NET_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#define NET_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#include "data/tensor.hpp"
#include "layer/abstract/non_param_layer.hpp"
#include "status_code.hpp"
namespace net_infer {
namespace activation {
/// 激活函数的类型别名，作用于两个张量（输入和输出）。
using ActivationFunc = std::function<void(sftensor, sftensor)>;

/// 支持的激活函数类型枚举。
enum class ActivationType {
  kActivatetionUnknown = -1,
  kActivationRelu = 0,
  kActivationSilu = 1,
  kActivationSigmoid = 2,
  kActivationHardSwish = 3,
  kActivationHardSigmoid = 4,
  kActivationRelu6 = 5,
};

/// 将 ActivationType 枚举值转换为其对应的字符串表示。
std::string ActivationTypeToString(ActivationType type);

/// 激活层，逐元素应用非线性激活函数。
/// 该层无可训练参数，支持多种激活类型。
class ActivationLayer : public NonParamLayer {
 public:
  explicit ActivationLayer(activation::ActivationType type, std::string layer_name);

  /// 在前向推理前校验输入和输出张量。
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  /// 将配置的激活函数应用于批次中的每个输入张量。
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

 private:
  ActivationType act_type_ = ActivationType::kActivatetionUnknown;
};
}  // namespace activation
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
