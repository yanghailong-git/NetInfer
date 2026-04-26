#ifndef NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_
#define NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_

#include "layer/abstract/param_layer.hpp"
#include "runtime/runtime_op.hpp"

namespace net_infer {

class BatchNorm2dLayer : public ParamLayer {
 public:
  explicit BatchNorm2dLayer(uint32_t num_features, float eps, std::vector<float> affine_weight,
                            std::vector<float> affine_bias);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& batch_layer);

 private:
  float eps_ = 1e-5f;
  std::vector<float> affine_weight_;
  std::vector<float> affine_bias_;
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_
