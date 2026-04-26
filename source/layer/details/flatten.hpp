#ifndef NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {
class FlattenLayer : public NonParamLayer {
 public:
  explicit FlattenLayer(int32_t start_dim, int32_t end_dim);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& flatten_layer);

 private:
  int32_t start_dim_ = 0;
  int32_t end_dim_ = 0;
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
