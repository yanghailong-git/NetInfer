#ifndef NET_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace net_infer {

/**
 * @brief 上采样模式枚举
 */
enum class UpSampleMode {
  kModeNearest = 0,
  kModeBilinear = 1,  // 目前上采样支持这两种
};

/**
 * @brief 上采样层（UpSample / Upsample）
 *
 * 支持最近邻（Nearest）和双线性（Bilinear）两种插值模式，
 * 可通过 align_corners 参数控制像素对齐方式。
 */
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
  float scale_h_ = 1.f;   // 高度方向缩放倍数
  float scale_w_ = 1.f;   // 宽度方向缩放倍数
  bool is_align_corner_ = false;  // 是否使用 align_corners 模式
  UpSampleMode mode_ = UpSampleMode::kModeNearest;  // 上采样插值模式
};

}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
