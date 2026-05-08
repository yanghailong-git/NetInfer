#ifndef NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_
#define NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_

#include "layer/abstract/param_layer.hpp"
#include "runtime/runtime_op.hpp"

namespace net_infer {

/// Batch Normalization layer for 2D feature maps (NCHW format).
/// Applies per-channel normalization using running mean and variance, followed by affine transformation.
class BatchNorm2dLayer : public ParamLayer {
 public:
  explicit BatchNorm2dLayer(uint32_t num_features, float eps, std::vector<float> affine_weight,
                            std::vector<float> affine_bias);

  /// Forward pass: normalizes each channel of the input and applies the learned affine scale and shift.
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /// Factory method to construct a BatchNorm2dLayer from a runtime operator.
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& batch_layer);

 private:
  float eps_ = 1e-5f;              ///< Small constant added to variance for numerical stability.
  std::vector<float> affine_weight_;  ///< Per-channel gamma (scale) parameters.
  std::vector<float> affine_bias_;    ///< Per-channel beta (shift) parameters.
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_
