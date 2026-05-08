#ifndef NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#define NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace net_infer {

// Flatten layer: reshapes a tensor by flattening dimensions from start_dim to end_dim
// Corresponds to PyTorch's torch.flatten operation
class FlattenLayer : public NonParamLayer {
 public:
  // Constructor: start_dim and end_dim define the range of dimensions to flatten
  explicit FlattenLayer(int32_t start_dim, int32_t end_dim);

  // Forward pass: flattens the specified dimension range of each input tensor
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  // Factory method: creates a FlattenLayer from a RuntimeOperator
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& flatten_layer);

 private:
  int32_t start_dim_ = 0;  // Starting dimension for flattening
  int32_t end_dim_ = 0;    // Ending dimension for flattening (inclusive)
};
}  // namespace net_infer
#endif  // NET_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
