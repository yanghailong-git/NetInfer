#ifndef NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
#define NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
#include "layer/abstract/param_layer.hpp"
namespace net_infer {

/**
 * @brief Matrix multiplication layer for LLaMA-style models.
 *
 * This layer performs a matrix multiplication between input tensors and
 * a stored weight tensor (w @ x). It supports both 1-D and 2-D input shapes.
 */
class LLamaMatmulLayer : public ParamLayer {
 public:
  /**
   * @brief Constructor.
   * @param weight_dim0 Number of rows in the weight matrix (output features).
   * @param weight_dim1 Number of columns in the weight matrix (input features).
   */
  explicit LLamaMatmulLayer(int32_t weight_dim0, int32_t weight_dim1);

  /**
   * @brief Forward pass: compute matrix multiplication for each input tensor.
   * @param inputs Input tensor batch.
   * @param outputs Output tensor batch (will be resized/created if empty).
   * @return Status code indicating success or failure.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  /**
   * @brief Set layer weights from a vector of shared tensor pointers.
   * @param weights Weight tensors (exactly one expected).
   */
  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

  /**
   * @brief Set layer weights from a flat float vector (not supported).
   * @param weights Flat weight data.
   */
  void set_weights(const std::vector<float>& weights) override;

 private:
  int32_t weight_dim0_ = 0;  ///< Output dimension (rows of weight matrix).
  int32_t weight_dim1_ = 0;  ///< Input dimension (columns of weight matrix).
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_MATMUL_HPP
