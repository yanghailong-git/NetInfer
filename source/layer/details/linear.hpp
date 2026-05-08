#ifndef NET_INFER_SOURCE_LAYER_LINEAR_HPP_
#define NET_INFER_SOURCE_LAYER_LINEAR_HPP_
#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace net_infer {

// Linear (fully connected) layer
// Computes the linear transformation: y = xW^T + b
// Corresponds to PyTorch's nn.Linear
class LinearLayer : public ParamLayer {
 public:
  //  explicit LinearLayer(uint32_t batch, uint32_t in_channel, uint32_t in_dim, uint32_t out_dim,
  //  bool use_bias = true);

  // Constructor: initializes the linear layer with input/output feature dimensions
  explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

  // Validates input, output, weight, and bias tensors
  StatusCode Check(const std::vector<sftensor>& inputs,
                   const std::vector<sftensor>& outputs) override;

  // Forward pass: computes linear transformation for each input tensor
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // Factory method: creates a LinearLayer from a RuntimeOperator
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& linear_layer);

  // Sets weights from a flat float vector
  void set_weights(const std::vector<float>& weights) override;

  // Sets weights from a vector of tensors (delegates to base class)
  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

 private:
  int32_t in_features_ = 0;   // Number of input features
  int32_t out_features_ = 0;  // Number of output features
  bool use_bias_ = false;     // Whether to use bias
};
}  // namespace net_infer

#endif  // NET_INFER_SOURCE_LAYER_LINEAR_HPP_
