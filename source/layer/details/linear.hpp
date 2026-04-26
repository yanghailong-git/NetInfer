#ifndef NET_INFER_SOURCE_LAYER_LINEAR_HPP_
#define NET_INFER_SOURCE_LAYER_LINEAR_HPP_
#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace net_infer {
class LinearLayer : public ParamLayer {
 public:
  //  explicit LinearLayer(uint32_t batch, uint32_t in_channel, uint32_t in_dim, uint32_t out_dim,
  //  bool use_bias = true);
  explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& linear_layer);

  void set_weights(const std::vector<float>& weights) override;

  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

 private:
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
  bool use_bias_ = false;
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_LINEAR_HPP_
