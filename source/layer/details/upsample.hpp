#ifndef NET_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace net_infer {
enum class UpSampleMode {
  kModeNearest = 0,
  kModeBilinear = 1,  // 目前上采样支持这两种
};

class UpSampleLayer : public NonParamLayer {
 public:
  explicit UpSampleLayer(float scale_h, float scale_w,
                         UpSampleMode mode = UpSampleMode::kModeNearest,
                         bool is_align_scale = false);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& upsample_layer);

 private:
  float scale_h_ = 1.f;
  float scale_w_ = 1.f;
  bool is_align_corner_ = false;
  UpSampleMode mode_ = UpSampleMode::kModeNearest;
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
