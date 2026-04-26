#ifndef NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
#define NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
#include "layer/abstract/param_layer.hpp"
#include "runtime/runtime_op.hpp"
namespace net_infer {
class RMSNormLayer : public ParamLayer {
 public:
  explicit RMSNormLayer();

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

  void set_weights(const std::vector<float>& weights) override;
 private:
  float eps_ = 1e-5f;
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
