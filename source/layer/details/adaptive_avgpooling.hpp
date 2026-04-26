#ifndef NET_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
#define NET_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {
class AdaptiveAveragePoolingLayer : public NonParamLayer {
 public:
  explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w);

  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& avg_layer);

 private:
  uint32_t output_h_ = 0;
  uint32_t output_w_ = 0;
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_AVGPOOLING_HPP_
