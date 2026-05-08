#ifndef NET_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#define NET_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#include "data/tensor.hpp"
#include "layer/abstract/non_param_layer.hpp"
#include "status_code.hpp"
namespace net_infer {
namespace activation {
/// Type alias for an activation function that operates on two tensors (input and output).
using ActivationFunc = std::function<void(sftensor, sftensor)>;

/// Enumeration of supported activation function types.
enum class ActivationType {
  kActivatetionUnknown = -1,
  kActivationRelu = 0,
  kActivationSilu = 1,
  kActivationSigmoid = 2,
  kActivationHardSwish = 3,
  kActivationHardSigmoid = 4,
  kActivationRelu6 = 5,
};

/// Converts an ActivationType enum value to its corresponding string representation.
std::string ActivationTypeToString(ActivationType type);

/// Activation layer that applies a non-linear activation function element-wise.
/// This layer has no trainable parameters and supports multiple activation types.
class ActivationLayer : public NonParamLayer {
 public:
  explicit ActivationLayer(activation::ActivationType type, std::string layer_name);

  /// Validates input and output tensors before forward inference.
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  /// Applies the configured activation function to each input tensor in the batch.
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

 private:
  ActivationType act_type_ = ActivationType::kActivatetionUnknown;
};
}  // namespace activation
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
