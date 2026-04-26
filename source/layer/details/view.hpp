#ifndef NET_INFER_SOURCE_LAYER_FLATTEN_HPP_
#define NET_INFER_SOURCE_LAYER_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {
class ViewLayer : public NonParamLayer {
 public:
  explicit ViewLayer(std::vector<int32_t> shapes);

  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& view_layer);

 private:
  std::vector<int32_t> shapes_;
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_FLATTEN_HPP_
