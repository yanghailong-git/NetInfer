#ifndef NET_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#define NET_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {
class ReluLayer : public activation::ActivationLayer {
 public:
  explicit ReluLayer();

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& relu_layer);
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
