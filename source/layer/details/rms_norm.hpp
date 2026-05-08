#ifndef NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
#define NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
#include "layer/abstract/param_layer.hpp"
#include "runtime/runtime_op.hpp"
namespace net_infer {

/**
 * @brief Root Mean Square (RMS) normalization layer.
 *
 * Normalizes input by dividing by the RMS of the input and then
 * applying a learnable scale (gain) parameter.
 * Formula: output = gain * (input / sqrt(mean(input^2) + eps)).
 */
class RMSNormLayer : public ParamLayer {
 public:
  explicit RMSNormLayer();

  /**
   * @brief Forward pass: apply RMS normalization to each input tensor.
   * @param inputs Input tensor batch.
   * @param outputs Output tensor batch (will be created if empty).
   * @return Status code indicating success or failure.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief Set layer weights from a vector of shared tensor pointers.
   * @param weights Weight tensors (exactly one gain tensor expected).
   */
  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

  /**
   * @brief Set layer weights from a flat float vector (not supported).
   * @param weights Flat weight data.
   */
  void set_weights(const std::vector<float>& weights) override;
 private:
  float eps_ = 1e-5f;  ///< Small constant added for numerical stability.
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_RMS_NORM_HPP
