#ifndef NET_INFER_SOURCE_LAYER_MAX_POOLING_
#define NET_INFER_SOURCE_LAYER_MAX_POOLING_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

/**
 * @brief 2-D max pooling layer.
 *
 * Extracts the maximum value from each pooling window sliding over the
 * spatial dimensions (height/width) of the input tensor.
 */
class MaxPoolingLayer : public NonParamLayer {
 public:
  /**
   * @brief Constructor.
   * @param padding_h Padding applied to the height dimension.
   * @param padding_w Padding applied to the width dimension.
   * @param pooling_size_h Height of the pooling window.
   * @param pooling_size_w Width of the pooling window.
   * @param stride_h Vertical stride of the pooling window.
   * @param stride_w Horizontal stride of the pooling window.
   */
  explicit MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h,
                           uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w);

  /**
   * @brief Forward pass: apply max pooling to each input tensor.
   * @param inputs Input tensor batch.
   * @param outputs Output tensor batch (will be created if empty).
   * @return Status code indicating success or failure.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief Validate input and output tensor arrays before forward.
   * @param inputs Input tensor batch.
   * @param outputs Output tensor batch.
   * @return Status code indicating success or failure.
   */
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  /**
   * @brief Factory method to create a MaxPoolingLayer from runtime operator parameters.
   * @param op Runtime operator containing "stride", "padding", and "kernel_size" parameters.
   * @param max_layer Output layer pointer to be assigned.
   * @return Status code indicating success or failure.
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& max_layer);

 private:
  uint32_t padding_h_ = 0;      ///< Height padding.
  uint32_t padding_w_ = 0;      ///< Width padding.
  uint32_t pooling_size_h_ = 0; ///< Height of the pooling kernel.
  uint32_t pooling_size_w_ = 0; ///< Width of the pooling kernel.
  uint32_t stride_h_ = 1;       ///< Vertical stride.
  uint32_t stride_w_ = 1;       ///< Horizontal stride.
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_MAX_POOLING_
